import math

import dgl
from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertConfig
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
import torch.nn as nn

class My_tokenizer(object):
    def __init__(self,config,vocab,max_seq_len,max_doc_len):
        self.config=config
        self.vocab=vocab
        self.max_seq_len=max_seq_len
        self.max_doc_len=max_doc_len
    def token(self,batch_data):
        #batch_data  [b_len,doc_len,seq_len,token_len]
        doc_seq_tok_ids=[]
        for b in batch_data:
            curb=[]
            for doc in b:
                curdoc=[]
                for seq in doc:
                    curseq=[]
                    for tok in seq:
                        curseq.append(self.vocab[tok])
                    if(len(curseq)<=self.max_seq_len):
                        curseq=curseq+['[PAD]' for _ in range(self.max_seq_len-len(curseq))]
                    else :
                        #这里只是简单做了截断处理 还可以处理更加仔细 就是如果一句话太长可以分为两句 第一句截断 第二句补充
                        # （多句话类比）
                        curseq=curseq[:self.max_seq_len]
                    curdoc.append(curseq)
                if(len(curdoc)<=self.max_doc_len):
                    curdoc=curdoc+[ ['[PAD]'] * self.max_seq_len for _ in range(self.max_doc_len-len(curdoc))]
                else :
                    # 这里只是简单做了截断处理
                    curdoc=curdoc[:self.max_doc_len]
                curb.append(curdoc)
            doc_seq_tok_ids.append(curb)
        return doc_seq_tok_ids
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel,self).__init__()
        #GCN模型 邻接矩阵需要保存
        #batch_doc --- 一个batch的数据集
        #tokenizer --- 将batch 的doc_seq_ids 转为 doc_seq_tok_ids
        #embedding --- 将batch 的doc_seq_tok_ids 转为 doc_seq_emb
        #bertmodel --- 将batch 的doc_seq_emb 转为 doc_seq_bert_emb
        #构建异构特征图 get_graph_from doc_seq_bert_emb(self):
        #GCNmodel --- 将 doc_graph  --- 转变为doc_graph_node_GCN_emb
        #FFmodel --- 将doc_graph_node_GCN_emb 中的相关事件映射为而分类问题
        #损失函数设计 较差熵损失或者nll_loss 或者mse 等等。
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.BertModel = BertModel.from_pretrained('bert-base-uncased')
    def prepare_data(self,batchsize,all_data):
        maxlen=0
        for i in range(len(all_data)):
            all_data[i] = bert_tokenizer.tokenize(''.join(all_data[i]))

            maxlen = max(maxlen, len(all_data[i]))
        docs_token_type_ids = []
        docs_attention_mask = []
        input_ids = []
        for seq in all_data:
            tseq = seq + ['[PAD]' for _ in range(maxlen - len(seq))]
            ids = 0
            curidslis = []
            curatmlis = []
            for t in tseq:
                curidslis.append(ids)
                if (t == '[SEP]'):
                    ids = 1
                if (t != '[PAD]'):
                    curatmlis.append(1)
                else:
                    curatmlis.append(0)
            docs_token_type_ids.append(curidslis)
            docs_attention_mask.append(curatmlis)
            print(tseq)
            print(curidslis)
            print(curatmlis)
            feature = bert_tokenizer.convert_tokens_to_ids(tseq)
            input_ids.append(feature)
            print(seq)
            print(feature)
        input_ids = torch.tensor(input_ids)
        docs_token_type_ids = torch.tensor(docs_token_type_ids)
        docs_attention_mask = torch.tensor(docs_attention_mask)
        return input_ids,docs_token_type_ids,docs_attention_mask
    def forward(self):
        pass

if __name__=="__main__":
    test_docs=[['[CLS]', 'this', 'is', 'blue', '[SEP]', 'that', 'is', 'red', '[SEP]'],
               ['[CLS]', 'that', 'is', 'red', '[SEP]', 'but', 'not', 'yellow', '[SEP]'],
               ['[CLS]','but','not','yellow','[SEP]','give','me','some','choice','[SEP]']]
    docs_token_type_ids=[]
    docs_attention_mask=[]
    input_ids=[]
    maxlen=0
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for i in range(len(test_docs)):
        test_docs[i] = bert_tokenizer.tokenize(''.join(test_docs[i]))

        maxlen = max(maxlen, len(test_docs[i] ))
    for seq in test_docs:
        tseq=seq+['[PAD]' for _ in range(maxlen-len(seq))]
        ids=0
        curidslis=[]
        curatmlis=[]
        for t in tseq:
            curidslis.append(ids)
            if(t=='[SEP]'):
                ids=1
            if(t!='[PAD]'):
                curatmlis.append(1)
            else:
                curatmlis.append(0)
        docs_token_type_ids.append(curidslis)
        docs_attention_mask.append(curatmlis)
        print(tseq)
        print(curidslis)
        print(curatmlis)
        feature = bert_tokenizer.convert_tokens_to_ids(tseq)
        input_ids.append(feature)
        print(seq)
        print(feature)
    bert_config=BertConfig()
    print(bert_config)
    BertModel = BertModel.from_pretrained('bert-base-uncased')
    input_ids=torch.tensor(input_ids)
    docs_token_type_ids=torch.tensor(docs_token_type_ids)
    docs_attention_mask=torch.tensor(docs_attention_mask)
    seq_pos_ids=torch.arange(
        3, dtype=torch.long, device='cpu', requires_grad=False
    ).unsqueeze(-1)
    # docs_seq_pos_ids=seq_pos_ids.unsqueeze(0).expand_as(input_ids)
    print(seq_pos_ids)
    docs_seq_pos_ids=torch.zeros(3,maxlen)
    div_term_sin=torch.exp(torch.arange(0.,maxlen,step=2,dtype=torch.long,requires_grad=False)*
                       -(math.log(10000.0)/maxlen))
    div_term_cos = torch.exp(torch.arange(1., maxlen, step=2, dtype=torch.long, requires_grad=False) *
                             -(math.log(10000.0) / maxlen))
    # print(div_term)
    # print(docs_seq_pos_ids[:,0::2])
    # print(seq_pos_ids*div_term)
    docs_seq_pos_ids[:,0::2]=torch.sin(seq_pos_ids*div_term_sin)
    docs_seq_pos_ids[:,1::2]=torch.cos(seq_pos_ids*div_term_cos)
    print(docs_seq_pos_ids)
    docs_seq_pos_ids=docs_seq_pos_ids.long()#为什么pos一定要是long类型的 感觉浮点数表达能力不是更强吗？
    print(docs_seq_pos_ids)
    docs_embedding=BertModel(input_ids=input_ids, token_type_ids=docs_token_type_ids,
                             position_ids=docs_seq_pos_ids,
                             attention_mask=docs_attention_mask,output_attentions=True)
    print(docs_embedding.attentions)
    print(docs_embedding.last_hidden_state.size())#3*13*768
    print(docs_embedding.pooler_output.size())#3*768
    print(docs_embedding.attentions[0].size())
    # print(BertModel)
    pass
