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

class mymodel(nn.Module):
    def __init__(self):
        super(mymodel,self).__init__()
        #GCN模型 邻接矩阵需要保存
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
    docs_embedding=BertModel(input_ids=input_ids, token_type_ids=docs_token_type_ids,
                             attention_mask=docs_attention_mask,output_attentions=True)
    print(docs_embedding.attentions)
    print(docs_embedding.last_hidden_state.size())#3*13*768
    print(docs_embedding.pooler_output.size())#3*768
    print(docs_embedding.attentions[0].size())
    # print(BertModel)
    pass
