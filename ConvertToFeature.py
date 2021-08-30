import re
import json
import os
import xml.dom.minidom
import logging
from ReadExample import DocExampleLoader

class DocFeatureConvert(object):
    def __init__(self,all_tok2ids,maxseqlen=10):
        self.all_tok2ids=all_tok2ids
        self.maxseqlen=maxseqlen
        #如果直接采用截断和补充的做法那么这样就可以了 还有一种做法就是先将长句子划分为多个短句子，然后重新改变每个tok的位置
        self.docid2docfeature=dict()

    def get_doc_feature(self,adoc):
        # print(adoc.seqids2toklis)
        doc_seq_tok_lis=adoc.seqids2toklis
        doc_len=len(doc_seq_tok_lis.keys())
        doc_feature_lis=[[] for _ in range(doc_len)]
        for k,v in doc_seq_tok_lis.items():
            for tok in v:
                doc_feature_lis[k].append(self.all_tok2ids[tok])
            if len(v)<=self.maxseqlen:
                doc_feature_lis[k]+=[len(self.all_tok2ids.keys()) for _ in range(self.maxseqlen-len(v))]
            else:
                doc_feature_lis[k]=doc_feature_lis[k][:self.maxseqlen]
        self.docid2docfeature[adoc.doc_id]=doc_feature_lis
    def __call__(self, doc_examples_dict,*args, **kwargs):
        for doc_id,adocexample in doc_examples_dict.items():
            self.get_doc_feature(adocexample)
        return self.docid2docfeature
        # for i in self.docid2docfeature[0]:
        #     print(i)

if __name__ == "__main__":
    pass
    # all_task_base_dir = os.getcwd()
    # docExampleLoader = DocExampleLoader(max_seq_len=10, max_doc_len=10, base_dir=os.path.join(all_task_base_dir,
    #                                                                                           'data/train/Causal-TimeBank-main'),
    #                                     train_flag=True,
    #                                     all_token_file=os.path.join(all_task_base_dir,
    #                                                                 'data/train/all_token.json'))
    # docid2docexample,docname2docid,all_token2ids,all_ids2token,docid2eventedocsigid2event=\
    #     docExampleLoader('cat/', 'timeml/')
    # docFeatureConvert=DocFeatureConvert(all_token2ids)
    # docid2docfeature=docFeatureConvert(docid2docexample)
