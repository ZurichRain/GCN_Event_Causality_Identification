# 定义一个model 类 用来 将特征最后变成 0/1 并根据所有事件构造正负样例

import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertConfig
from ReadExample import DocExampleLoader
from ConvertToFeature import DocFeatureConvert
import os
import math
import dgl.nn.pytorch as dglnn
import dgl

class MyEmbedding(nn.Module):
    def __init__(self,config,vocab,hidden_size=512,dropout=0.1):
        super(MyEmbedding, self).__init__()
        self.config=config
        self.vocab=vocab
        self.vocab_len=len(vocab)+1
        self.hidden_size=hidden_size
        self.emb_layer=nn.Embedding(self.vocab_len,self.hidden_size)
        self.pos_emb_layer=nn.Embedding(self.config.seq_len,self.hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,doc_seq_tok_ids):
        batch_size,doc_len, seq_len =doc_seq_tok_ids.size()
        doc_seq_tok_emb=self.emb_layer(doc_seq_tok_ids)
        seq_tok_pos_ids=torch.arange(0,self.config.seq_len,dtype=torch.long,requires_grad=False)

        doc_seq_tok_pos_ids=seq_tok_pos_ids.unsqueeze(0).unsqueeze(0).\
            expand(batch_size,doc_len,seq_len)

        doc_seq_pos_emb=self.pos_emb_layer(doc_seq_tok_pos_ids)

        doc_seq_tok_emb_all=doc_seq_tok_emb+doc_seq_pos_emb
        doc_seq_tok_emb_out=self.layer_norm(doc_seq_tok_emb_all)
        doc_seq_tok_emb_out=self.dropout(doc_seq_tok_emb_out)
        return doc_seq_tok_emb_out

def Attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = [nn.Linear(d_model, d_model) for _ in range(4)]
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = Attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)

class MyFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model,ff_size, dropout=0.1):
        super(MyFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, ff_size)
        self.w_2 = nn.Linear(ff_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class MyBertEncodeModel(nn.Module):
    def __init__(self, config,vocab,emb_size=512,hidden_size=768
                 ,ff_size=512,head_size=3,dropout=0.1,encode_layer_num=1):
        #config 中需要保存一个seqlen
        super(MyBertEncodeModel, self).__init__()
        self.config = config
        self.encode_layer_num=encode_layer_num
        self.hidden_size=hidden_size
        self.ff_size=ff_size
        self.head_size=head_size
        self.emb_size=emb_size
        self.vocab=vocab
        self.emb_layer=MyEmbedding(self.config,vocab,self.emb_size)
        self.liner_model=nn.Linear(emb_size,hidden_size)

        self.multiHeadedAttention_lis=[MultiHeadedAttention(head_size, hidden_size, dropout=dropout)
                                       for _ in range(self.encode_layer_num)]
        self.ff_model_lis=[MyFeedForward(hidden_size,ff_size,dropout=dropout)
                           for _ in range(self.encode_layer_num)]
    def forward(self,doc_seq_tok):
        batch_size, doc_len, seq_len = doc_seq_tok.size()
        doc_seq_tok_emb = self.emb_layer(doc_seq_tok)
        doc_seq_tok_emb=self.liner_model(doc_seq_tok_emb)
        doc_seq_tok_att_emb = self.multiHeadedAttention_lis[0](doc_seq_tok_emb, doc_seq_tok_emb, doc_seq_tok_emb)
        doc_seq_tok_att_emb = doc_seq_tok_att_emb.view(batch_size, doc_len, seq_len, self.hidden_size)
        doc_seq_tok_att_emb_out = self.ff_model_lis[0](doc_seq_tok_att_emb)
        doc_seq_tok_emb_out=doc_seq_tok_emb+doc_seq_tok_att_emb_out
        for i in range(1,self.encode_layer_num):
            doc_seq_tok_att_emb=self.multiHeadedAttention[i](doc_seq_tok_emb_out,doc_seq_tok_emb_out,
                                                          doc_seq_tok_emb_out)
            doc_seq_tok_att_emb=doc_seq_tok_att_emb.view(batch_size,doc_len,seq_len,self.hidden_size)
            doc_seq_tok_att_emb_out=self.ff_model[i](doc_seq_tok_att_emb)
            doc_seq_tok_emb_out = doc_seq_tok_emb_out + doc_seq_tok_att_emb_out
        return doc_seq_tok_emb_out

class MyTask(object):
    def __init__(self,config,vocab,batch_data,docid2docfeature,docid2eventedocsigid2event
                 ,epoch_num=10,encod_hidden_size=768,GCN_hidden_size=512):
        #确定网络结构所必须的所有超参数 包括模型
        #第一层首先需要编码
        self.config=config
        self.vocab=vocab
        self.batch_data=batch_data
        self.docid2docfeature=docid2docfeature
        self.docid2eventedocsigid2event=docid2eventedocsigid2event
        self.encod_hidden_size=encod_hidden_size
        self.GCN_hidden_size=GCN_hidden_size
        self.bert_encode_layer=MyBertEncodeModel(self.config,self.vocab,hidden_size=self.encod_hidden_size)
        #结束之后变成了 batch_len,doc_len,seq_len,hidden_size
        #接下来就是构建图神经网络
        self.GCN_layer = dglnn.GraphConv(in_feats=self.encod_hidden_size,out_feats=self.GCN_hidden_size,
                                         weight=True, bias=True,allow_zero_in_degree=True)
        self.full_concat=nn.Linear(2*self.GCN_hidden_size,1,bias=True)
        self.softmax_layer=nn.Softmax(dim=-1)
        # self.loss_func = torch.nn.CrossEntropyLoss()
        # self.loss_func = torch.nn.NLLLoss()

        # self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD([
                {'params': self.bert_encode_layer.parameters()},
                {'params': self.GCN_layer.parameters()},
                {'params': self.full_concat.parameters()}
            ], lr=0.1)
        self.epoch_num=epoch_num
        # 对于每个事件id --》 GCNemb表示后 对俩俩不同的事件进行拼接，经过一层全连接层 就可以判断这两个事件之间是否有关系
        #这里需要建立一个一一对应的关系  for i in range(maxid) for j in range(maxid) if(i!=j) ff(concat(get(i),get(j)))


    def get_graph_from_doc_seq_emb(self,batch_data,batch_emb):
        #对于节点的特征可以通过该batch的数据中mention所在的位置求平均进行编码
        #节点之间的关系可以通过这个doc里的mention和普通实体 建立边关系，普通实体和普通实体之间 mention和mention之间
        #在这个文档里没有标注普通实体，仅标注了事件，所以直接对所有事件mention建立图关系
        # print(batch_emb.size())
        n_event=len(batch_data.doc_events)
        u=[]
        v=[]
        for eidxi in range(n_event):
            for eidxj in range(n_event):
                if (eidxj == eidxi):
                    continue
                u.append(eidxi)
                v.append(eidxj)
        u, v = torch.tensor(u), torch.tensor(v)
        g = dgl.graph((u, v))
        node_f=[]
        eventedocsigid2event=self.docid2eventedocsigid2event[batch_data.doc_id]
        for eidx in range(n_event):
            cur_event=eventedocsigid2event[eidx]
            tok_ids_span=cur_event.event_tok_span
            event_tok_emb_lis=[]
            for seqid,tokid in tok_ids_span:
                event_tok_emb_lis.append(batch_emb[seqid][tokid].unsqueeze(0))
            event_tok_emb=torch.cat(event_tok_emb_lis,dim=0)
            # print(event_tok_emb)
            event_emb=torch.mean(event_tok_emb,dim=0)
            # print(event_emb)
            node_f.append(event_emb.unsqueeze(0))
        node_f=torch.cat(node_f,dim=0)
        # print(node_f.size())
        g.ndata['x']=node_f
        # g = dgl.add_self_loop(g) #添加自环
        return g
    def train(self):
        for epoch in range(self.epoch_num):
            for bidx in range(len(self.batch_data)):
                # print(bidx)
                cur_batch_data=self.batch_data[bidx]
                print(len(cur_batch_data.doc_event_relations))
                cur_batch_data_f=self.docid2docfeature[cur_batch_data.doc_id]
                cur_batch_data_f=torch.tensor(cur_batch_data_f)
                cur_batch_data_f=cur_batch_data_f.unsqueeze(0) #这里是我构造了一个doc看作一个batch
                cur_batch_data_emb=self.bert_encode_layer(cur_batch_data_f)
                g=self.get_graph_from_doc_seq_emb(cur_batch_data,cur_batch_data_emb[0])
                g_emb=self.GCN_layer(g,g.ndata['x'])
                all_concat_lis=[]
                for eidxi in range(len(cur_batch_data.doc_events)):
                    for eidxj in range(len(cur_batch_data.doc_events)):
                        if(eidxj==eidxi):
                            continue
                        if((eidxi,eidxj) in cur_batch_data.doc_event_relations_edoc_sig_id_pair):
                            concat_ij=torch.cat([g_emb[eidxi],g_emb[eidxj]],dim=-1)
                            all_concat_lis.append(concat_ij.unsqueeze(0))
                if(len(all_concat_lis)<1):
                    continue
                all_concat_f=torch.cat(all_concat_lis,dim=0)
                all_concat_out=self.full_concat(all_concat_f)
                # p_label=self.softmax_layer(all_concat_out)
                print(all_concat_out)
                p_label=F.log_softmax(all_concat_out, dim=-1)
                m_label=torch.argmax(p_label,dim=-1)
                t_label=torch.tensor(cur_batch_data.label_lis,dtype=torch.long)
                # print(m_label)
                # print(t_label)
                loss=-torch.mean(p_label)
                # loss=self.loss_func(p_label,t_label)
                print("bidx",bidx)
                print("loss:",loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


class Config(object):
    def __init__(self,config_dict):
        for k,v in config_dict.items():
            setattr(self,k,v)
if __name__ == "__main__":
    maxseqlen=100
    # atest=torch.tensor([
    #     [
    #         [2349, 7957, 6379, 5272, 15, 5635, 11, 8150, 6385, 8029],
    #         [2836, 4991, 3416, 5129, 7957, 3047, 1112, 3294, 7957, 1853],
    #         [1815, 7957, 5722, 8154, 5101, 5400, 15, 7957, 8293, 6290],
    #         [2836, 1827, 4257, 5292, 5867, 8154, 7494, 6517, 6290, 5638],
    #         [1815, 2700, 15, 7728, 5369, 3081, 5017, 8524, 5874, 19],
    #         [1815, 7957, 2451, 15, 3081, 5101, 8524, 5874, 19, 8543],
    #         [1010, 5463, 1766, 1979, 15, 3081, 7997, 6517, 4589, 19],
    #         [2220, 6744, 5463, 1766, 1979, 5066, 3081, 6570, 15, 5066],
    #         [1744, 11, 924, 11, 1910, 2017, 19, 8543, 8543, 8543],
    #         [2317, 7956, 5846, 3212, 15, 3551, 7957, 1289, 7886, 15]
    #      ],
    #     [
    #         [2711, 8422, 3041, 1993, 7625, 7390, 6044, 4548, 5066, 7986],
    #         [1731, 7559, 7957, 6798, 8029, 5017, 3737, 3294, 7261, 5300],
    #         [1815, 1766, 1979, 15, 5625, 3267, 3566, 8029, 7957, 7349],
    #         [2322, 8462, 6198, 4101, 8192, 8389, 15, 7997, 6290, 5365],
    #         [1010, 3041, 1993, 8448, 3534, 5889, 8029, 3677, 4785, 19],
    #         [983, 3141, 1766, 1979, 15, 7957, 6798, 5955, 5292, 4214],
    #         [2400, 2394, 6436, 4649, 4056, 5337, 19, 8543, 8543, 8543],
    #         [2394, 3740, 6147, 4996, 3737, 19, 8543, 8543, 8543, 8543],
    #         [2688, 4771, 5334, 6797, 15, 8506, 3081, 5412, 7984, 6044],
    #         [2845, 3562, 3219, 5635, 8448, 3534, 3267, 8268, 5211, 8268]
    #
    #     ],
    #     [
    #         [1218, 5463, 7957, 6486, 7997, 6100, 15, 7728, 5296, 6603],
    #         [1771, 3367, 6331, 7984, 6517, 5136, 19, 8543, 8543, 8543],
    #         [3069, 3762, 5183, 7390, 6517, 4480, 3434, 7957, 7437, 5926],
    #         [2857, 7991, 4547, 6507, 5066, 4649, 5412, 4548, 19, 8543],
    #         [919, 8158, 5412, 4547, 8484, 5640, 5066, 4974, 4548, 19],
    #         [2762, 15, 7967, 3367, 4967, 3737, 19, 8543, 8543, 8543],
    #         [1010, 3434, 7957, 3589, 3705, 5402, 15, 3199, 7929, 8528],
    #         [1796, 7975, 7956, 7957, 6102, 5625, 4912, 5200, 15, 3294],
    #         [2711, 5066, 1766, 1979, 15, 5635, 11, 8025, 15, 3393],
    #         [2867, 4654, 5371, 6320, 8017, 6361, 5183, 6385, 15, 3393]
    #     ]
    # ])
    # print(atest.size())
    adict={'seq_len':maxseqlen}
    test_conf=Config(adict)
    all_task_base_dir = os.getcwd()
    docExampleLoader = DocExampleLoader(max_seq_len=maxseqlen, max_doc_len=10, base_dir=os.path.join(all_task_base_dir,
                                                                                              'data/train/Causal-TimeBank-main'),
                                        train_flag=True,
                                        all_token_file=os.path.join(all_task_base_dir,
                                                                    'data/train/all_token.json'))
    docid2docexample, docname2docid, all_token2ids, all_ids2token, docid2eventedocsigid2event = \
        docExampleLoader('cat/', 'timeml/')
    docFeatureConvert = DocFeatureConvert(all_token2ids,maxseqlen=maxseqlen)
    docid2docfeature = docFeatureConvert(docid2docexample)
    batch_data = []
    idx=0
    for k,v in docid2docexample.items():
        # if(idx>3):
        #     break
        batch_data.append(v)
        idx+=1
    atask=MyTask(test_conf,all_token2ids,batch_data,docid2docfeature,docid2eventedocsigid2event)
    atask.train()
    # amodel=MyBertEncodeModel(test_conf,all_token2ids)
    # atest_encod=amodel(atest)
    # print(atest_encod.size())
