
#前提是首先构造 seq_tok=[] 这个可以通过xml文件构造
#

# doc_example  -- > [seq_example]
# seq_example -- > [word_example]
# word_example -- > [tok_example]

# doc_feature -- > [seq_tok_ids]
#

class BaseToken(object):
    #会产生一个token_text 一样但是对象不一样
    def __init__(self,tok_text,tok_gid,seq_id=-1,seq_inner_id=-1):
        self.tok_text=tok_text
        self.tok_gid=tok_gid
        self.seq_id=seq_id
        self.seq_inner_id=seq_inner_id
        self.tok_span=[seq_id,seq_inner_id]

    def update_by_dict(self, data_dict):
        for k, v in data_dict.items():
            setattr(self, k, v)

    def update_by_kv(self, k, v):
        setattr(self, k, v)
    def __repr__(self):
        s=""
        s+='''
    BaseToken(
        tok_text = {}
        tok_gid = {}
        seq_id = {}
        seq_inner_id = {}
    )
    '''.format(self.tok_text,self.tok_gid,self.seq_id,self.seq_inner_id)
        return s


class Word(object):
    def __init__(self,word_text,word_id):
        self.word_text=word_text
        self.word_id=word_id
        self.tok_lis=[]
        self.word_span=[]
    def update_by_dict(self,data_dict):
        for k,v in data_dict.items():
            setattr(self,k,v)
    def update_by_kv(self,k,v):
        setattr(self,k,v)
    def add_tok(self,tok):
        self.tok_lis.append(tok)

    def get_event_span(self):
        seq_id,tok_s_ids=self.tok_lis[0].tok_span
        seq_id,tok_e_ids=self.tok_lis[-1].tok_span
        self.word_span.append(seq_id)
        self.word_span.append(tok_s_ids)
        self.word_span.append(tok_e_ids)

class Seqexample(object):
    def __init__(self,seq_id,seq_text,seq_toks):
        self.seq_id=seq_id
        self.seq_text=seq_text
        self.seq_toks=seq_toks
        self.seq_ws=seq_text.split(' ')
    def update_by_dict(self,data_dict):
        for k,v in data_dict.items():
            setattr(self,k,v)
    def update_by_kv(self,k,v):
        setattr(self,k,v)

    def update_tok_seq_innerid_and_seq_id(self):
        for idx,tok in enumerate(self.seq_toks):
            tok.update_by_kv('seq_id',self.seq_id)
            tok.update_by_kv('seq_inner_id',idx)
    def __repr__(self):
        s=""
        s+='''
        Seqexample(
            seq_id = {}
            seq_text = {}
            seq_toks = {}
            seq_ws = {}
        )
        '''.format(self.seq_id,self.seq_text,self.seq_toks,self.seq_ws)
        return s

class Event(object):
    def __init__(self,aspect="",certainty="",class_="",
                 comment="",factuality="",id="",edoc_sig_id=-1,modality="",polarity="",pos="",tense=""):
        self.aspect=aspect
        self.certainty=certainty
        self.class_=class_
        self.comment=comment
        self.factuality=factuality
        self.id=id
        self.edoc_sig_id=edoc_sig_id
        self.modality=modality
        self.polarity=polarity
        self.pos=pos
        self.tense=tense
        self.tok_lis=[]
        self.tok_ids_lis=[]
        self.event_tok_span=[]

    def add_tok(self,tok):
        self.tok_lis.append(tok)

    # def get_event_span(self):
    #     seq_id,tok_s_ids=self.tok_lis[0].tok_span
    #     seq_id,tok_e_ids=self.tok_lis[-1].tok_span
    #     self.event_span.append(seq_id)
    #     self.event_span.append(tok_s_ids)
    #     self.event_span.append(tok_e_ids)

    def update_by_dict(self,data_dict):
        for k,v in data_dict.items():
            setattr(self,k,v)
    def update_by_kv(self,k,v):
        setattr(self,k,v)

    def __repr__(self):
        s=""
        s+='''
Event(
    aspect = {}
    certainty = {}
    class_ = {}
    comment = {}
    factuality = {}
    id = {}
    modality = {}
    polarity = {}
    pos = {}
    tense = {}
    tok_lis = {}
    tok_ids_lis = {}
    event_tok_span = {}
)'''.format(self.aspect,self.certainty,self.class_,self.comment,self.factuality
                   ,self.id,self.modality,self.polarity,self.pos,self.tense,self.tok_lis,
            self.tok_ids_lis,self.event_tok_span)
        return s

class EventRelation(object):
    def __init__(self,relation_id=-1,s_event=None,e_event=None):
        self.relation_id=relation_id
        self.s_event=s_event
        self.e_event=e_event
    def __repr__(self):
        s=""
        s+='''
        EventRelation(
            relation_id = {}
            s_event = {}
            e_event = {}
        )
        '''.format(self.relation_id,self.s_event,self.e_event)
        return s

class Docsample(object):
    def __init__(self,doc_text="",doc_id=-1):
        self.doc_text=doc_text
        self.doc_id=doc_id
        self.doc_seqs=doc_text.split('\n')
        #一个事件中包含子事件怎么搞
        self.doc_events=[]
        self.doc_event_relations=[]
        self.doc_event_relations_edoc_sig_id_pair=[]
        self.doc_seqs_ids=[]
        self.seqids2toklis=dict()
        self.label_lis=[]
    def update_by_dict(self,data_dict):
        for k,v in data_dict.items():
            setattr(self,k,v)
    def update_by_kv(self,k,v):
        setattr(self,k,v)
    def add_event(self,event):
        self.doc_events.append(event)
    def add_relation(self,relation):
        self.doc_event_relations.append(relation)

    def __repr__(self):
        s=""
        s+='''
Docsample(
    doc_text = {}
    doc_id = {}
    doc_seqs = {}
    doc_event_relations = {}
    doc_seqs_ids = {}
    seqids2toklis = {}
)
        '''.format(self.doc_text,self.doc_id,self.doc_seqs,self.doc_event_relations,self.doc_seqs_ids,
                   self.seqids2toklis)
        return s

if __name__ == "__main__":
    a=BaseToken('ddd',1,0,1)
    b=Event()
    b.add_tok(a)
    print(b)
