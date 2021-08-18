import numpy as np
import pandas as pd
import os
import xml.dom.minidom
import json
import re, collections
import matplotlib.pyplot as plt

class myError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) #初始化父类
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo
class fileiotool(object):
    def __init__(self):
        pass
    def write_2Dmatrix_to_txt(self,data,file_path,data_encoding='utf-8'):
        with open(file_path,mode='w',encoding=data_encoding) as f:
            for i in data:
                f.writelines(' '.join(i)+ '\n')
    def write_to_txt(self,data,file_path,data_encoding='utf-8'):
        with open(file_path,mode='w',encoding=data_encoding) as f:
            for i in data:
                f.writelines(str(i)+'\n')
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
def parse_label(fa):
    text_data=""
    for i in fa.childNodes:
        if(i.nodeName!='#text'):
            text_data+=parse_label(i)
            # text_data+= ' '
        else :
            text_data+=i.data
    return text_data
def get_doc_seq_txt(dom_tree):
    root = dom_tree.documentElement
    TEXTs = root.getElementsByTagName('TEXT')
    doc_text = ""
    for text in TEXTs:
        for j in text.childNodes:
            if (j.nodeName == '#text'):
                doc_text += j.data
                # doc_text += ' '
            elif (j.nodeName == 'EVENT'):
                doc_text+=parse_label(j)
                # doc_text += ' '
            elif (j.nodeName == 'TIMEX3'):
                doc_text+=parse_label(j)
                # doc_text += ' '
            elif (j.nodeName == 'C-SIGNAL'):
                doc_text+=parse_label(j)
                # doc_text += ' '
            else :
                raise myError("不知道的node类型:{}".format(j.nodeName))
    doc_text = doc_text.strip()
    return doc_text.split('\n')
def get_seq_tokenwords(xml_dom_tree):
    root = xml_dom_tree.documentElement
    Tokens=root.getElementsByTagName('token')
    seq2tokenlis=dict()
    for token in Tokens:
        seqid=token.getAttribute('sentence')
        cur_token=""
        for j in token.childNodes:
            cur_token+=j.data
        if (seq2tokenlis.get(seqid)):
            seq2tokenlis[seqid].append(cur_token)
        else:
            seq2tokenlis[seqid]=[]
            seq2tokenlis[seqid].append(cur_token)
    return seq2tokenlis

def reset_seqlen(doc_seq_lis,maxlen):
    # token lis
    res_doc_seq_lis=[]
    for seqlis in doc_seq_lis:
        newcurseq=[]
        if(len(seqlis)<maxlen):
            newcurseq=seqlis
            while(len(newcurseq)<maxlen):
                newcurseq.append('unk')
            res_doc_seq_lis.append(newcurseq)
        else:
            idx=0
            while(idx<len(seqlis)):
                if(len(newcurseq)==maxlen):
                    res_doc_seq_lis.append(newcurseq)
                    newcurseq=[]
                newcurseq.append(seqlis[idx])
                idx+=1
            while(len(newcurseq)<maxlen):
                newcurseq.append('unk')
            res_doc_seq_lis.append(newcurseq)

    return res_doc_seq_lis

def reset_doc_len(doc_seq_tokens,maxlen):
    res=doc_seq_tokens
    if(len(doc_seq_tokens)<maxlen):
        while(len(doc_seq_tokens)<maxlen):
            res.append(['unk']*len(doc_seq_tokens[0]))
    else:
        res=res[:maxlen]
    return res
if __name__=='__main__':
    io_tool=fileiotool()
    base_path=os.getcwd()
    tml_file_dir=os.path.join(base_path,'data/train/Causal-TimeBank-main/timeml/')
    name2doc_seq_txt=dict()
    for tml_file in os.listdir(tml_file_dir):
        if(tml_file[-3:]!='tml'):
            continue
        doc_seq_txt=io_tool.get_data_from_tml(os.path.join(tml_file_dir,tml_file),get_doc_seq_txt)
        name2doc_seq_txt[tml_file[:-3]+'xml']=doc_seq_txt
    xml_file_dir=os.path.join(base_path,'data/train/Causal-TimeBank-main/cat/')
    tot_len=0
    tot_s=0
    maxlens=0
    minlens=24
    len2nums=dict()
    token2idx=io_tool.get_data_from_json(os.path.join(base_path,'data/train/all_token.json'))
    token2idx['unk']=len(token2idx.keys())
    # print(token2idx)
    for xml_file in os.listdir(xml_file_dir):
        if(xml_file[-3:]!='xml'):
            continue
        curdocseqtxt=name2doc_seq_txt[xml_file]
        tot_s+=len(curdocseqtxt)
        if (len2nums.get(len(curdocseqtxt))):
            len2nums[len(curdocseqtxt)] += 1
        else:
            len2nums[len(curdocseqtxt)] = 1
        curseq_token_lis=io_tool.get_data_from_xml(os.path.join(xml_file_dir,xml_file),get_seq_tokenwords)
        doc_seq_tokens=[]
        for i in range(len(curdocseqtxt)):
            curlis=curseq_token_lis[str(i)]
            tot_len+=len(curlis)
            maxlens=max(maxlens,len(curlis))
            minlens=min(minlens,len(curlis))
            doc_seq_tokens.append(curlis)
            seqs=""
            for j in curlis:
                seqs+=j
                seqs+=' '
            seqs=seqs[:-1]
            # print(seqs, "------", curdocseqtxt[i])
            assert re.sub(' ','',seqs) == re.sub(' ','',curdocseqtxt[i])
        doc_reset_seq_tokens=reset_seqlen(doc_seq_tokens,45)
        doc_reset_seq_tokensidx=[]
        for i in doc_reset_seq_tokens:
            curseq_tokensidx=[]
            for j in i:
                curseq_tokensidx.append(str(token2idx[j]))
            doc_reset_seq_tokensidx.append(curseq_tokensidx)
        # io_tool.write_2Dmatrix_to_txt(doc_reset_seq_tokens,
        #                               os.path.join(base_path,'data/train/seq_reset_len_tokens/'+
        #                                            xml_file[:-4]+'_seq_tokens_matrix.txt'))
        # io_tool.write_2Dmatrix_to_txt(doc_reset_seq_tokensidx,
        #                               os.path.join(base_path, 'data/train/seq_reset_len_tokensidx/' +
        #                                            xml_file[:-4] + '_seq_tokensidx_matrix.txt'))
        # for i in doc_reset_seq_tokens:
        #     print(i)
        # print(doc_seq_tokens)
    #一个文档中最长是有75句话 最短是有2句话
    #平均文档中有27句话
    #有92.3%的文档中句子个数是小于35的

    #有两句话是一个词 可以考虑直接删除
    #最长句子为78， 最短句子为1
    #平均句子长度是:24.069706103993973
    #有95.8%的句子长度都是小于45的
    # print(tot_len*1.0/tot_s)
    # print(maxlens)
    # print(minlens)
    # len2nums=sorted(len2nums.items(), key=lambda e: e[0])
    # # ans=0
    # # for i in len2nums:
    # #     if(i[0]<45):
    # #         ans+=i[1]
    # # print(ans*1.0/tot_s)
    # x=[i[0] for i in len2nums]
    # y=[i[1] for i in len2nums]
    # ans=0
    # for i in len2nums:
    #     if(i[0]<35):
    #         ans+=i[1]
    # print(ans*1.0/np.sum(y))
    # print(len2nums)
    # plt.plot(x,y)
    # plt.show()
