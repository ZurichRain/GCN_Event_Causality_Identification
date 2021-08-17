import dgl
import transformers
import logging
import random
import os
import json
import sys
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn.parallel as para
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
import  xml.dom.minidom

logging.basicConfig(level=logging.DEBUG) #用来配置logging的输出等级 默认是error等级
# DEBUG < INFO < WARNING < ERROR < CRITICAL

class myError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) #初始化父类
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo

class fileiotool(object):
    def __init__(self):
        pass
    def write_to_json(self,data_dict,file_path,data_encoding="utf-8",ensure_ascii=False,indent=4,**kwargs):
        with open(file_path,mode='w',encoding=data_encoding) as f:
            json.dump(data_dict,f,ensure_ascii=ensure_ascii,indent=indent,**kwargs)
    def get_data_from_json(self,file_path,data_encoding="utf-8"):
        assert os.path.exists(file_path)
        with open(file_path,mode='r',encoding=data_encoding) as f:
            data=json.load(f)
        return data
    def get_data_from_tml(self,file_path,parse_tml_fuc):
        assert os.path.exists(file_path)
        curdom= xml.dom.minidom.parse(file_path)
        data = parse_tml_fuc(curdom)
        return data
    def get_data_from_xml(self,file_path,parse_xml_fuc):
        assert os.path.exists(file_path)
        curdom= xml.dom.minidom.parse(file_path)
        data = parse_xml_fuc(curdom)
        return data
#tasksetting
class Tasksetting(object):
    base_default_sets=[
        ("bert_model","bert-base-chinese"),
        ("train_dir","train/"),
        ("dev_dir","dev/"),
        ("test_dir","test/"),
        ("max_seq_len",128),
        ("data_dir","data/"),
        ("model_dir","model/"),
        ("output_dir","output/"),
        ("no_cuda",True),
        ("xml_file_dir",'Causal-TimeBank-main/cat/'),
        ("tml_file_dir",'Causal-TimeBank-main/timeml/')
    ]
    iotool = fileiotool()
    def __init__(self,add_config_dict=dict()):
        for attr,val in Tasksetting.base_default_sets:
            setattr(self,attr,val)
        for k,v in add_config_dict.items():
            setattr(self,k,v)
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
    def update_by_dict(self,config_dict=dict()):
        for k,v in config_dict.items():
            setattr(self,k,v)
    def dump_to(self,dir_path,filename="base_task_setting.json",**kwargs):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filepath=os.path.join(dir_path,filename)
        self.iotool.write_to_json(self.__dict__,filepath,**kwargs)

#实现一个tasking 基类 其中包括了整个任务的流程
class Basetask(object):
    def __init__(self,setting=None):
        assert setting!=None
        self.setting=setting
        self.logger=logging.getLogger(self.__class__.__name__)
        self.init_random_seed()
    def logging(self,msg,level=logging.INFO):
        self.logger.log(level=level,msg=msg)
    def init_random_seed(self,seed=None):
        #初始化所有的seed
        if seed==None:
            # print(self.setting.__dict__)
            assert self.setting.__dict__.get('seed')
            seed=self.setting.seed
        self.logging("="*20+"reset seed to {}".format(seed)+"="*20)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if not self.setting.no_cuda:
            torch.cuda.manual_seed(seed)
    def load_example_feature_dataset(self,load_example_obj,convert_to_feature_obj,convert_to_dataset_obj,
                                     file_path=None):
        assert file_path is not None ,("数据集的路径不合法或者是没有设置数据集的路径")
        assert os.path.exists(file_path),"您所配置的数据集文件不存在"
        self.logging("="*20+"load example from {}".format(file_path)+"="*20)
        examples=load_example_obj(file_path)
        self.logging("="*20+"convert feature from {}".format(file_path)+"="*20)
        features=convert_to_feature_obj(examples)
        self.logging("=" * 20 + "convert dataset from {}".format(file_path) + "=" * 20)
        datasets=convert_to_dataset_obj(features)
        return examples,features,datasets

    def _load_data(self,load_example_obj,convert_to_feature_obj,convert_to_dataset_obj
                   ,is_train=True,is_dev=True,is_test=True,
                   train_file_path=None,dev_file_path=None,test_file_path=None):
        ##这里是加载训练测试数据集的地方
        #抽象化为对不同的子任务可以通过传递不同类型的函数来达到加载数据的目的
        self.logging("="*20+"load task data"+"="*20)
        data_dict=dict()
        if is_train:
            data_dict['train_sample'],data_dict['train_feature'],data_dict['train_dataset']=self.load_example_feature_dataset(
                load_example_obj,convert_to_feature_obj,convert_to_dataset_obj,
                file_path=train_file_path
            )
        if is_dev:
            data_dict['dev_sample'], data_dict['dev_feature'], data_dict['dev_dataset'] = self.load_example_feature_dataset(
                load_example_obj, convert_to_feature_obj, convert_to_dataset_obj,
                file_path=dev_file_path
            )
        if is_test:
            data_dict['test_sample'], data_dict['test_feature'], data_dict['test_dataset'] = self.load_example_feature_dataset(
                load_example_obj, convert_to_feature_obj, convert_to_dataset_obj,
                file_path=test_file_path
            )
        return data_dict

#sample和feature之间的区别是 feature是规整的 而sample仅包含原来的样本信息，只不过是包装起来方便以后处理
class Seqsample(object):
    #一句话也需要保存很多信息，
    #句子全局idx，句子所在的文档idx，句子所在文档中的idx，句子所包含的token，句子所包含的tokenidx
    def __init__(self,Seqsample_config):
        for k,v in Seqsample_config:
            setattr(self,k,v)
    def update_attr_by_dict(self,attr_dict=dict()):
        for k,v in attr_dict.items():
            setattr(self,k,v)
    def update_attr_by_kv(self,k,v):
        setattr(self,k,v)

class Docsample(object):
    def __init__(self,Docsample_config=dict()):
        #文档级别的sample所包含的属性包括：
        #文档唯一id，文档中包含的句子个数seq_num，每一句话中的token列表seq_token[seq_len,token_len]
        for k,v in Docsample_config:
            setattr(self,k,v)
    def update_attr_by_dict(self,attr_dict=dict()):
        for k,v in attr_dict.items():
            setattr(self,k,v)
    def update_attr_by_kv(self,k,v):
        setattr(self,k,v)

class Load_example(object):
    def __init__(self,event_dict,token_dict,st_event_idx,token2idx,doc_id):
        self.event_dict=event_dict
        self.token_dict=token_dict
        self.st_event_idx=st_event_idx
        self.token2idx=token2idx
        self.doc_id=doc_id
    def get_subwords(self,doc_text,tot_subwords_lis,curidx):

        pass
    def get_subwords2tokensidx(self):
        pass
    def get_subwords2seqidx(self):
        pass
    def represent_doc(self,doc_text,subword2idx):
        doc_lis=[]
        for seq in doc_text:
            curseqlis=[]
            for w in seq.split(" "):
                curs=""
                for i in w:
                    curs+=i
                    if(curs in subword2idx.keys()):
                        curseqlis.append(curs)
                        curs=""
                if(len(curs)):
                    curseqlis.append(curs)
            doc_lis.append(curseqlis)
        return doc_lis
    def parse_tml_fuc(self,dom_tree):
        #需要返回一个Docsample对象
        res_Docsample=Docsample()
        root=dom_tree.documentElement
        TEXTs=root.getElementsByTagName('TEXT')
        doc_text=""
        eventlist=[]
        #这里是获取文档内容
        for i in TEXTs:
            for j in i.childNodes:
                if(j.nodeName=='#text'):
                    doc_text+=j.data
                elif(j.nodeName=='EVENT'):
                    eventlist.append(j)
                    cur_event_id=j.getAttribute('eid')[1:]
                    cur_event=self.event_dict[cur_event_id]
                    print('=' * 20 + 'event(事件)' + '=' * 20)
                    for k,v in j.attributes.items():
                        print(k,"----->",v)
                    curevent_str=""
                    for k in j.childNodes:
                        doc_text+=k.data
                        curevent_str+=k.data
                    cur_event.update_attr_by_kv('data',curevent_str)
                    cur_event_token_list=[]
                    jiaoyan_str=""
                    for k in cur_event.token_idx_list:
                        cur_event_token_list.append(self.token_dict[k])
                        jiaoyan_str+=self.token_dict[k].data
                    assert jiaoyan_str==curevent_str
                    cur_event.update_attr_by_kv('token_list',cur_event_token_list)
                    cur_event.update_attr_by_kv('seq_idx',self.token_dict[cur_event.token_idx_list[0]].sentence)
                    cur_event.update_attr_by_kv('doc_id',self.doc_id)
                    self.event_dict[cur_event_id]=cur_event
                    print(curevent_str)
                elif(j.nodeName=='TIMEX3'):
                    print('='*20+'time(时间)'+'='*20)
                    for k,v in j.attributes.items():
                        print(k,"----->",v)
                    for k in j.childNodes:
                        doc_text+=k.data
                elif(j.nodeName=='C-SIGNAL'):
                    print('=' * 20 + 'c-signal(原因触发词)' + '=' * 20)
                    for k, v in j.attributes.items():
                        print(k, "----->", v)
                    for k in j.childNodes:
                        print(k.data)
                        doc_text += k.data
                else :
                    raise myError("不知道的node类型:{}".format(j.nodeName))
        all_idx_token_dict = dict()
        for k, v in self.token_dict.items():
            cur_token_str = v.data
            # v.update_attr_by_kv('event_idx',)
            v.update_attr_by_kv('tot_idx', self.token2idx[cur_token_str])
            all_idx_token_dict[self.token2idx[cur_token_str]] = v
        all_idx_event_dict=dict()
        cur_event_idx2totidx=dict()
        for k,v in self.event_dict.items():
            for j in v.token_list:
                all_idx_token_dict[self.token2idx[j.data]].update_attr_by_kv('event_idx',self.st_event_idx)
                j.update_attr_by_kv('tot_idx', self.token2idx[j.data])
            cur_event_idx2totidx[v.id]=self.st_event_idx
            all_idx_event_dict[self.st_event_idx]=v
            self.st_event_idx+=1


        doc_text=doc_text.strip()
        res_Docsample.update_attr_by_kv('doc_text_str',doc_text)
        doc_text_seq_list=doc_text.split('\n')
        res_Docsample.update_attr_by_kv('seq_text_list', doc_text_seq_list)
        res_Docsample.update_attr_by_kv('event_dict', all_idx_event_dict)
        res_Docsample.update_attr_by_kv('token_dict', all_idx_token_dict)
        res_Docsample.update_attr_by_kv('doc_id',self.doc_id)

        instance_list=root.getElementsByTagName('MAKEINSTANCE')
        for i in instance_list:
            #eventID属性:
            #eiid属性：
            #aspect属性: 事件时态的补充 ie<完成时，进行时>
            #tense属性: 事件的时态<过去时，当前时>
            #polarity属性: 事件属性是积极还是消极
            #pos属性: 事件的词性<动词，名词>
            print('=' * 20 + 'instance' + '=' * 20)
            for k,v in i.attributes.items():
                print(k,"----->",v)
        tlink_list=root.getElementsByTagName('TLINK')
        for i in tlink_list:
            #tid属性：
            #relType属性： 时态依赖类型<before,after> 表示s事件在e事件之前发生还是之后发生还有同时发生
            #eventInstanceID属性： 事件实例的id <eiid>
            #relatedToTime属性： 被依赖的时间属性 比如:事件i在时间j之前发生
            #relatedToEventInstance属性：被依赖的事件属性 比如:事件i在事件j之前发生
            print('=' * 20 + 't-link' + '=' * 20)
            for k,v in i.attributes.items():
                print(k, "----->", v)
        clink_list=root.getElementsByTagName('CLINK')
        label_list=[]
        # label2idx=dict()
        for i in clink_list:
            print('=' * 20 + 'c-link' + '=' * 20)
            cur_label_dict=dict()
            for k,v in i.attributes.items():
                if(k=='eventInstanceID' or k=='relatedToEventInstance'):
                    v=cur_event_idx2totidx[v[2:]]
                cur_label_dict[k]=v
            cur_tag=Label(cur_label_dict)
            label_list.append(cur_tag)
            for k, v in i.attributes.items():
                print(k, "----->", v)
        for i in doc_text_seq_list:
            print(i)
        for i in label_list:
            print(i)
        # print(len(doc_text_seq_list))
        res_Docsample.update_attr_by_kv('gold_label_list', label_list)
        #get label 的时候 我需要获取s和t 和它们之间的依赖关系
        return res_Docsample

    def load_example_fuc(self,file_path):
        # 这里应该实现批处理
        curiotool = fileiotool()
        example = curiotool.get_data_from_tml(file_path, self.parse_tml_fuc)
        return example
    def __call__(self, filepath ,*args, **kwargs):
        example=self.load_example_fuc(filepath)
        return example

class GCNtask(Basetask):
    def __init__(self,gcn_config_setting):

        super(GCNtask,self).__init__(gcn_config_setting)
        self.event_all_idx=1
        curiotool = fileiotool()
        tot_tokens=[]
        cur_dir=os.getcwd()
        train_xml_file_path=os.path.join(cur_dir,self.setting.data_dir,self.setting.train_dir
                                         ,self.setting.xml_file_dir)
        train_tml_file_path = os.path.join(cur_dir, self.setting.data_dir, self.setting.train_dir,
                                           self.setting.tml_file_dir)
        xml_file_list=os.listdir(train_xml_file_path)
        tml_file_list=os.listdir(train_tml_file_path)
        xml_file_list=sorted(xml_file_list)
        tml_file_list=sorted(tml_file_list)
        xml_file_list=xml_file_list[1:]
        tml_file_list=tml_file_list[1:]
        for xml_file in xml_file_list:
            # if(xml_file[-3:]!='xml'):
            #     continue
            xml_file_path=os.path.join(train_xml_file_path,xml_file)
            cur_tot_tokens = curiotool.get_data_from_xml(xml_file_path, get_token_from_xml)
            tot_tokens+=cur_tot_tokens
        tot_tokens=list(set(tot_tokens))
        self.token2idx=dict()
        self.idx2token=dict()
        for idx,val in enumerate(tot_tokens):
            self.token2idx[val]=idx
            self.idx2token[idx]=val
        #loop
        event_dict, token_dict = self.get_event_info_from_xml(os.path.join(train_xml_file_path, xml_file_list[0])
                                                              , curiotool)
        self.load_example_obj = Load_example(event_dict, token_dict,self.event_all_idx,self.token2idx,1)
        cur_doc_sample_data = self.load_example_obj(os.path.join(train_tml_file_path, tml_file_list[0]))

        #将每个doc 转变为一个 tokenidx list maxlen  根据maxlen 进行切割
        self.convert_to_feature_obj = Convert_to_feature()

        self.convert_to_dataset_obj = Convert_to_dataset()
        for idx,xml_file in enumerate(xml_file_list):
            event_dict,token_dict=self.get_event_info_from_xml(train_xml_file_path,curiotool)
            self._load_data(self.load_example_obj,self.convert_to_feature_obj,self.convert_to_dataset_obj,
                        train_file_path=train_tml_file_path)
    def parse_event_info_from_xml(self,dom_tree):
        root = dom_tree.documentElement
        all_event_dict = dict()
        EVENTs = root.getElementsByTagName('EVENT')
        for i in EVENTs:
            #每个事件要有个全局的idx
            #self.event_all_idx=1
            # print(i.attributes)
            cureventid=i.getAttribute('id')
            all_event_dict[cureventid]=Event()
            # self.event_all_idx+=1
            for k,v in i.attributes.items():
                all_event_dict[cureventid].update_attr_by_kv(k,v)
            #处理token
            cur_token_list=[]#token id 仅仅是当前文档的id
            for j in i.childNodes:
                if(j.nodeName=='token_anchor'):
                    for k,v in j.attributes.items():
                        cur_token_list.append(v)
            all_event_dict[cureventid].update_attr_by_kv('token_idx_list', cur_token_list)
        Tokens_list = root.getElementsByTagName('token')
        all_token_dict=dict()
        for i in Tokens_list:
            cur_token_id=i.getAttribute('id')
            all_token_dict[cur_token_id] = GCNTokens()
            token_data=""
            for j in i.childNodes:
                token_data+=j.data

            all_token_dict[cur_token_id].update_attr_by_kv('data',token_data)
            for k,v in i.attributes.items():
                all_token_dict[cur_token_id].update_attr_by_kv(k,v)

        return all_event_dict ,all_token_dict
            # cur_events_dict['']

    def get_event_info_from_xml(self,xml_file_path,curiotool):
        data=curiotool.get_data_from_xml(xml_file_path,self.parse_event_info_from_xml)
        return data

#这里希望能把事件抽象成为一个类
class Event(object):
    #首先要有一个idx--->tokenidx的映射 还要有个反映射
    #id-->事件本身的映射 同理反映射
    #
    def __init__(self,attr_dict=dict()):
        self.aspect=None
        self.certainty=None
        self.class_=None
        self.comment=None
        self.factuality=None
        self.eid=None
        self.modality=None
        self.polarity=None
        self.pos=None
        self.tense=None
        for k,v in attr_dict.items():
            setattr(self,k,v)
    def update_attr_by_dict(self,attr_dict=dict()):
        for k,v in attr_dict.items():
            setattr(self,k,v)
    def update_attr_by_kv(self,k,v):
        setattr(self,k,v)
class GCNTokens(object):
    #首先要有一个idx--->tokenidx的映射 还要有个反映射
    #id-->事件本身的映射 同理反映射
    #
    def __init__(self,attr_dict=dict()):
        self.id=None
        self.data=None
        self.seq_idx=None
        self.event_idx=None
        for k,v in attr_dict.items():
            setattr(self,k,v)
    def update_attr_by_dict(self,attr_dict=dict()):
        for k,v in attr_dict.items():
            setattr(self,k,v)
    def update_attr_by_kv(self,k,v):
        setattr(self,k,v)

class Label(object):
    #sid,eid,sspan,espan 一定是两个事件
    #s_seq_id,e_seq_id,
    def __init__(self,attr_dict=dict()):
        for k,v in attr_dict.items():
            setattr(self,k,v)
    def update_attr_by_dict(self,attr_dict=dict()):
        for k,v in attr_dict.items():
            setattr(self,k,v)
    def __repr__(self):
        s="Label:\n"
        for k,v in self.__dict__.items():
            s+=k+'-'*10+'>'+str(v)
            s+='\n'
        return s

#这里仅用来测试数据是否能够读入到对象中


def parse_xml_fuc(dom_tree,**kwargs):
    root=dom_tree.documentElement
    TEXTs=root.getElementsByTagName('TEXT')
    doc_text=""
    eventlist=[]
    #这里是获取文档内容
    token2idx=None
    if(kwargs.get('token2idx')):
        token2idx=kwargs['token2idx']
    for i in TEXTs:
        for j in i.childNodes:
            if(j.nodeName=='#text'):
                doc_text+=j.data
            elif(j.nodeName=='EVENT'):
                eventlist.append(j)
                print('=' * 20 + 'event(事件)' + '=' * 20)
                for k,v in j.attributes.items():
                    print(k,"----->",v)
                curevent=""
                for k in j.childNodes:
                    doc_text+=k.data
                    curevent+=k.data
                print(curevent)
                #idx
                if token2idx is not None:
                    token_idx=token2idx[curevent]
            elif(j.nodeName=='TIMEX3'):
                print('='*20+'time(时间)'+'='*20)
                for k,v in j.attributes.items():
                    print(k,"----->",v)
                for k in j.childNodes:
                    doc_text+=k.data
            elif(j.nodeName=='C-SIGNAL'):
                print('=' * 20 + 'c-signal(原因触发词)' + '=' * 20)
                for k, v in j.attributes.items():
                    print(k, "----->", v)
                for k in j.childNodes:
                    print(k.data)
                    doc_text += k.data
            else :
                raise myError("不知道的node类型:{}".format(j.nodeName))

    doc_text=doc_text.strip()
    doc_text_seq_list=doc_text.split('\n')

    instance_list=root.getElementsByTagName('MAKEINSTANCE')
    for i in instance_list:
        #eventID属性:
        #eiid属性：
        #aspect属性: 事件时态的补充 ie<完成时，进行时>
        #tense属性: 事件的时态<过去时，当前时>
        #polarity属性: 事件属性是积极还是消极
        #pos属性: 事件的词性<动词，名词>
        print('=' * 20 + 'instance' + '=' * 20)
        for k,v in i.attributes.items():
            print(k,"----->",v)
    tlink_list=root.getElementsByTagName('TLINK')
    for i in tlink_list:
        #tid属性：
        #relType属性： 时态依赖类型<before,after> 表示s事件在e事件之前发生还是之后发生还有同时发生
        #eventInstanceID属性： 事件实例的id <eiid>
        #relatedToTime属性： 被依赖的时间属性 比如:事件i在时间j之前发生
        #relatedToEventInstance属性：被依赖的事件属性 比如:事件i在事件j之前发生
        print('=' * 20 + 't-link' + '=' * 20)
        for k,v in i.attributes.items():
            print(k, "----->", v)
    clink_list=root.getElementsByTagName('CLINK')
    label_list=[]
    # label2idx=dict()
    for i in clink_list:
        print('=' * 20 + 'c-link' + '=' * 20)
        cur_tag=Label(i.attributes)
        label_list.append(cur_tag)
        for k, v in i.attributes.items():
            print(k, "----->", v)
    for i in doc_text_seq_list:
        print(i)
    for i in label_list:
        print(i)
    # print(len(doc_text_seq_list))
    #get label 的时候 我需要获取s和t 和它们之间的依赖关系


def get_token_from_xml(dom_tree):
    #此处是实现得到一个dom中的所有token
    doc_tot_tokens=[]
    root = dom_tree.documentElement
    tokens=root.getElementsByTagName('token')
    for i in tokens:
        for j in i.childNodes:
            doc_tot_tokens.append(j.data)
    return doc_tot_tokens

def load_example_fuc(file_path,**kwargs):
    #这里应该实现批处理
    curiotool=fileiotool()
    cur_tot_tokens = curiotool.get_data_from_xml(kwargs['xml_file_path'], get_token_from_xml)
    print(cur_tot_tokens)
    ori_data=curiotool.get_data_from_xml(file_path,parse_xml_fuc)
    # print(ori_data)

    pass
def convert_to_feature_fuc():
    pass
def convert_to_dataset_fuc():
    pass



if __name__ == "__main__":
    # my_base_task_setting=tasksetting()
    # ex_config=dict()
    # ex_config['no_cuda']=True
    # ex_config['GCN_layer_num']=3
    # ex_config['seed']=1024
    # my_base_task_setting.update_by_dict(ex_config)
    # my_base_task=basetask(my_base_task_setting)
    # load_example_fuc('./data/Causal-TimeBank-main/timeml/wsj_0026.tml',
# xml_file_path="./data/Causal-TimeBank-main/cat/wsj_0026.xml")
    my_base_task_setting = Tasksetting()
    exconfig=dict()
    exconfig['no_cuda']=True
    exconfig['seed']=1024
    my_base_task_setting.update_by_dict(exconfig)
    my_GCN_task=GCNtask(my_base_task_setting)
