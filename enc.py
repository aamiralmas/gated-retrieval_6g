import pickle, numpy as np, re
_d=pickle.load(open('./prep.pkl','rb')); items=_d['items']; tf=_d['tf']; svd=_d['svd']; u=_d['u']; cnt=_d['cnt']; tot=_d['tot']; voc=_d['voc']
_tok=re.compile(r"[a-z0-9]+"); _a=1e-3
def _n(Z): return (Z/np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-9)).astype('float32')
def lsa_raw(texts): return svd.transform(tf.transform(texts))
def lsa(texts): return _n(lsa_raw(texts))
def sif_raw(texts):
    out=np.zeros((len(texts),100))
    for i,t in enumerate(texts):
        ws=[w for w in _tok.findall(t.lower()) if w in voc]
        if ws: out[i]=np.sum([voc[w]*(_a/(_a+cnt[w]/tot)) for w in ws],axis=0)
    return out-np.outer(out@u,u)          # linear map (sum, then projection)
def sif(texts): return _n(sif_raw(texts))
ENC={'A':(lsa,lsa_raw),'B':(sif,sif_raw)}

# ---- Encoder C: sparse lexical (tf-idf, uni+bigrams, linear tf => additive before normalisation) ----
import scipy.sparse as sp, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
_cp='./encC.pkl'
if os.path.exists(_cp): tfC=pickle.load(open(_cp,'rb'))
else:
    tfC=TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1,2),min_df=1,sublinear_tf=False,norm=None).fit([i['p'] for i in items]); pickle.dump(tfC,open(_cp,'wb'))
def lex_raw(texts): return tfC.transform(texts)
def lex(texts): return normalize(lex_raw(texts),norm='l2',axis=1).astype('float32').tocsr()
ENC['C']=(lex,lex_raw)
