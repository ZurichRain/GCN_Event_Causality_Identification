import re
import json
import os
import xml.dom.minidom
import logging
from BaseType import *

logging.basicConfig(level=logging.DEBUG) #用来配置logging的输出等级 默认是error等级
# DEBUG < INFO < WARNING < ERROR < CRITICAL

class MyError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) #初始化父类
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo

class FileIoTool(object):
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
    def get_data_from_tml(self,file_path,parse_tml_fuc,*args,**kwargs):
        assert os.path.exists(file_path)
        curdom= xml.dom.minidom.parse(file_path)
        data = parse_tml_fuc(curdom,*args,**kwargs)
        return data
    def get_data_from_xml(self,file_path,parse_xml_fuc,*args,**kwargs):
        assert os.path.exists(file_path)
        curdom= xml.dom.minidom.parse(file_path)
        data = parse_xml_fuc(curdom,*args,**kwargs)
        return data

class DocExampleLoader(object):
    def __init__(self,max_seq_len,max_doc_len,base_dir,train_flag,all_token_file=None):
        self.max_seq_len=max_seq_len
        self.max_doc_len=max_doc_len
        self.base_dir=base_dir
        self.train_flag=train_flag
        self.all_token_file=all_token_file
        self.curdocids=0
        self.docname2docid=dict()
        self.docid2docexample=dict()
        self.io_tool=FileIoTool()
        self.docid2seqids2toklis=dict()
        self.docid2wordids2tokdocidslis=dict()

    def parse_xml_get_tokens(self,dom_tree):
        root = dom_tree.documentElement
        tokens = root.getElementsByTagName('token')
        res_tokens=[]
        for token in tokens:
            tok_txt=""
            for child in token.childNodes:
                tok_txt+=child.data
            res_tokens.append(tok_txt)
            tok_seq_ids = int(token.getAttribute('sentence'))
            if (self.docid2seqids2toklis[self.curdocids].get(tok_seq_ids)):
                self.docid2seqids2toklis[self.curdocids][tok_seq_ids].append(tok_txt)
            else:
                self.docid2seqids2toklis[self.curdocids][tok_seq_ids]=[tok_txt]
        return res_tokens

    def get_all_tokens(self,xml_file_dir):
        xml_file_dir=os.path.join(self.base_dir,xml_file_dir)
        all_xml_tokens=[]
        for xml_file_name in os.listdir(xml_file_dir):
            if(xml_file_name[-3:]!='xml'):
                continue
            self.docid2seqids2toklis[self.curdocids]=dict()
            self.docname2docid[xml_file_name[:-4]]=self.curdocids
            xml_file = os.path.join(xml_file_dir,xml_file_name)
            all_xml_tokens += self.io_tool.get_data_from_xml(xml_file,self.parse_xml_get_tokens)
            self.curdocids += 1
        if self.train_flag:
            all_xml_tokens=sorted(list(set(all_xml_tokens)))
            self.all_token2ids=dict()
            self.all_ids2token = dict()
            for ids , tok in enumerate(all_xml_tokens):
                self.all_token2ids[tok]=ids
                self.all_ids2token[ids]=tok
        else:
            if self.all_token_file is not None :
                self.all_token2ids=self.io_tool.get_data_from_json(self.all_token_file)
                self.all_ids2token = dict()
                for k,v in self.all_token2ids.items():
                    self.all_ids2token[v]=k
            else:
                raise MyError("对于测试集应该提供token文件")

    def parse_node(self,curnode):
        cur_text=""
        for child in curnode.childNodes:
            if (child.nodeName != '#text'):
                #这里可能需要一个全局idx 和 seq idx
                cur_text += self.parse_node(child)
            else:
                cur_text += child.data
        return cur_text
    def parse_tml_get_doc(self,dom_tree,curdoc):
        root = dom_tree.documentElement
        texts = root.getElementsByTagName('TEXT')
        for text in texts:
            curdoc.doc_text+=self.parse_node(text)
        curdoc.doc_text=curdoc.doc_text.strip()
        curdoc.update_by_kv('doc_seqs',curdoc.doc_text.split('\n'))

    def get_all_doc(self,tml_file_dir):
        tml_file_dir = os.path.join(self.base_dir, tml_file_dir)
        for tml_file_name in os.listdir(tml_file_dir):
            if(tml_file_name[-3:]!='tml'):
                continue
            cur_doc_id=self.docname2docid[tml_file_name[:-4]]
            adoc=Docsample(doc_id=cur_doc_id)
            tml_file=os.path.join(tml_file_dir,tml_file_name)
            self.io_tool.get_data_from_tml(tml_file,self.parse_tml_get_doc,curdoc=adoc)
            # print(adoc)
            # print(self.docid2seqids2toklis[cur_doc_id])
            self.docid2docexample[cur_doc_id]=adoc

    def get_docid_wordids_tokdocidslis(self):
        for docid,docexample in self.docid2docexample.items():
            self.docid2wordids2tokdocidslis[docid]=dict()
            doc_text_lis=docexample.doc_seqs
            pre_w_len=0
            pre_tok_len=0
            docexample.doc_seqs_ids.append(pre_tok_len)
            for seqids,seq in enumerate(doc_text_lis):
                w_lis=seq.strip().split(' ')
                tok_lis=self.docid2seqids2toklis[docid][seqids]
                curtokids=0
                curtok_s=""
                for wseqids,w in enumerate(w_lis):
                    wids=pre_w_len+wseqids
                    self.docid2wordids2tokdocidslis[docid][wids]=[]
                    w=w.strip()
                    while(curtokids<len(tok_lis) and curtok_s!=w):
                        self.docid2wordids2tokdocidslis[docid][wids].append(pre_tok_len+curtokids)
                        curtok_s+=tok_lis[curtokids]
                        curtokids+=1
                    curtok_s=""
                pre_w_len+=len(w_lis)
                pre_tok_len+=len(tok_lis)
                docexample.doc_seqs_ids.append(pre_tok_len)
        # print(self.docid2docexample[0])
        # print(self.docid2seqids2toklis[0])
        # print(self.docid2wordids2tokdocidslis[0])

    def parse_xml_get_event(self,dom_tree,curdoc):
        root = dom_tree.documentElement
        Events = root.getElementsByTagName('EVENT')
        for event in Events:
            aevent=Event()
            aevent.aspect=event.getAttribute('aspect')
            aevent.certainty = event.getAttribute('certainty')
            aevent.class_=event.getAttribute('class')
            aevent.comment=event.getAttribute('comment')
            aevent.factuality = event.getAttribute('factuality')
            aevent.id = event.getAttribute('id')
            aevent.modality = event.getAttribute('modality')
            aevent.polarity = event.getAttribute('polarity')
            aevent.pos = event.getAttribute('pos')
            aevent.tense = event.getAttribute('tense')
            for tok in event.getElementsByTagName('token_anchor'):
                tok_ids=int(tok.getAttribute('id'))-1
                for xids,x in enumerate(curdoc.doc_seqs_ids):
                    if(tok_ids<x):
                        tok_seq_ids=tok_ids-curdoc.doc_seqs_ids[xids-1]
                        aevent.tok_lis.append(self.docid2seqids2toklis[curdoc.doc_id][xids-1][tok_seq_ids])
                        break
                aevent.tok_ids_lis.append(tok_ids)
            curdoc.doc_events.append(aevent)
            # if(len(aevent.tok_ids_lis)>1):
            #     print(aevent)

    def get_doc_event_from_xml(self,xml_file_dir):
        #实现了每个文档的事件添加
        xml_file_dir = os.path.join(self.base_dir, xml_file_dir)
        for xml_file_name in os.listdir(xml_file_dir):
            if (xml_file_name[-3:] != 'xml'):
                continue
            xml_file=os.path.join(xml_file_dir,xml_file_name)
            curdocid=self.docname2docid[xml_file_name[:-4]]
            adoc=self.docid2docexample[curdocid]
            self.io_tool.get_data_from_xml(xml_file,self.parse_xml_get_event,curdoc=adoc)

        # print(self.docid2docexample[0].doc_events)
        # print("*"*100)
        # print(self.docname2docid['ABC19980114.1830.0611'])
        # print(self.docid2docexample[135].doc_events)


    def __call__(self,xml_json_dir,tml_json_dir,*args, **kwargs):
        self.get_all_tokens(xml_json_dir)
        #下一步就是 构建每个doc_sample
        self.get_all_doc(tml_json_dir)
        self.get_docid_wordids_tokdocidslis()
        #还需要知道每个事件以及事件之间的关系
        self.get_doc_event_from_xml(xml_json_dir)


if __name__ == "__main__":
    # print(os.getcwd())
    all_task_base_dir=os.getcwd()

    # os.getcwd()
    docExampleLoader=DocExampleLoader(max_seq_len=10,max_doc_len=10,base_dir=os.path.join(all_task_base_dir,
                                        'data/train/Causal-TimeBank-main'),train_flag=True,
                                      all_token_file=os.path.join(all_task_base_dir,
                                        'data/train/all_token.json'))
    docExampleLoader('cat/','timeml/')
