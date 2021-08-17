import numpy as np
import pandas as pd
import os
import xml.dom.minidom
import json
import re, collections

class myError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) #初始化父类
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo
class fileiotool(object):
    def __init__(self):
        pass
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
        else :
            text_data+=i.data
    return text_data
def get_doc_words(dom_tree):
    root = dom_tree.documentElement
    TEXTs = root.getElementsByTagName('TEXT')
    doc_text = ""
    for text in TEXTs:
        for j in text.childNodes:
            if (j.nodeName == '#text'):
                doc_text += j.data
            elif (j.nodeName == 'EVENT'):
                doc_text+=parse_label(j)
            elif (j.nodeName == 'TIMEX3'):
                doc_text+=parse_label(j)
            elif (j.nodeName == 'C-SIGNAL'):
                doc_text+=parse_label(j)
            else :
                raise myError("不知道的node类型:{}".format(j.nodeName))
    doc_text = doc_text.strip()
    doc_text = doc_text.lower()
    r = "[,.$#%@^&*!;()\n]"
    doc_text = re.sub(r, '', doc_text)
    doc_text = re.sub('--','',doc_text)
    return doc_text.split(' ')

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


if __name__ == "__main__" :
    curdir=os.getcwd()
    xml_file_dir=os.path.join(curdir,'data/train/Causal-TimeBank-main/cat/')
    tml_file_dir=os.path.join(curdir,'data/train/Causal-TimeBank-main/timeml/')
    io_tool=fileiotool()
    all_words_dict=dict()
    for tml_file in os.listdir(tml_file_dir):
        if(tml_file[-3:]!= 'tml'):
            continue
        curdoc_words=io_tool.get_data_from_tml(os.path.join(tml_file_dir,tml_file),get_doc_words)
        for w in curdoc_words:
            if ' '.join(w)+' </w>' not in all_words_dict.keys():
                all_words_dict[' '.join(w)+' </w>']=1
            else:
                all_words_dict[' '.join(w)+' </w>']+=1
    # print(all_words_dict)
    num_merges = 1000
    for i in range(num_merges):
        pairs = get_stats(all_words_dict)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        all_words_dict = merge_vocab(best, all_words_dict)
        print(best)
    all_words_dict_lis=sorted(all_words_dict.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)
    all_words_dict=dict()
    all_subwords_lis=set()
    for i in all_words_dict_lis:
        k,v=i
        for s in k.split(' '):
            all_subwords_lis.add(s)
        all_words_dict[k]=[]
        all_words_dict[k].append(v)
        all_words_dict[k].append(re.sub(' ','',k))
    print(all_subwords_lis)
    io_tool.write_to_txt(all_subwords_lis,'subwords.txt')
    io_tool.write_to_json(all_words_dict,'words2subwords.json')

    print(all_words_dict)
    #遍历每一个
