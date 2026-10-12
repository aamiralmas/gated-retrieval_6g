"""Build corpora and embeddings. Everything derives from public data; seeded."""
import json, gzip, re, numpy as np, pickle, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
SEED=20261011; rs=np.random.RandomState(SEED)
# ---- TeleQnA ----
tq=json.load(open('data/TeleQnA.txt'))
items=[]
for k,v in tq.items():
    ans=re.sub(r'^option \d+:\s*','',v['answer']); akey=re.match(r'option (\d+)',v['answer']).group(1)
    wrong=[v[o] for o in v if o.startswith('option') and o!='option '+akey and v[o]]
    items.append(dict(src='tq',cat=v['category'],q=v['question'],p=ans+'. '+v.get('explanation',''),wrong=wrong))
# ---- ORAN-Bench-13K ----
for f in 'EMH':
    for l in open('data/ORAN-Bench-13K/Benchmark/fin_%s.json'%f):
        l=l.strip()
        if not l: continue
        q,opts,a=json.loads(l.rstrip(',')); opts=[re.sub(r'^\d+\.\s*','',o) for o in opts]; a=int(a)-1
        if not (0<=a<len(opts)) or not opts[a].strip(): continue
        items.append(dict(src='ob',cat='ORAN-'+f,q=q,p=opts[a],wrong=[o for i,o in enumerate(opts) if i!=a]))
# drop exact-duplicate passages/questions so that "own passage" ground truth is unambiguous
seenp=set(); seenq=set(); keep=[]
for it in items:
    kp=it['p'].strip().lower(); kq=it['q'].strip().lower()
    if kp in seenp or kq in seenq: continue
    seenp.add(kp); seenq.add(kq); keep.append(it)
items=keep; N=len(items); print('items',N, {s:sum(i['src']==s for i in items) for s in ('tq','ob')})
P=[i['p'] for i in items]; Q=[i['q'] for i in items]
# ---- Encoder A: LSA-256 (linear tf so the encoder is additive in term counts) ----
t0=time.time()
tf=TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1,1),min_df=2,sublinear_tf=False,norm=None)
Xp=tf.fit_transform(P)   # vocabulary/idf from the knowledge base only (index-time fit)
svd=TruncatedSVD(n_components=768,random_state=SEED,algorithm='randomized',n_iter=7).fit(Xp)
def lsa(texts):
    Z=svd.transform(tf.transform(texts)).astype('float32'); n=np.linalg.norm(Z,axis=1,keepdims=True); return Z/np.maximum(n,1e-9)
PA=lsa(P); QA=lsa(Q); print('lsa',time.time()-t0, PA.shape)
# ---- Encoder B: GloVe-100 SIF ----
voc={}; 
with gzip.open('data/glove-wiki-gigaword-100.gz','rt',encoding='utf8') as f:
    first=f.readline().split()
    if len(first)!=2: voc[first[0]]=np.array(first[1:],dtype='float32')
    for l in f:
        a=l.rstrip().split(' ')
        if len(a)==101: voc[a[0]]=np.asarray(a[1:],dtype='float32')
print('glove',len(voc))
words=list(voc); G=np.stack([voc[w] for w in words]); np.save('glove400k.npy',G)
tok=re.compile(r"[a-z0-9]+")
from collections import Counter
cnt=Counter(w for t in P+Q for w in tok.findall(t.lower())); tot=sum(cnt.values()); a=1e-3
def sif_raw(texts):
    out=np.zeros((len(texts),100),dtype='float32')
    for i,t in enumerate(texts):
        ws=[w for w in tok.findall(t.lower()) if w in voc]
        if ws: out[i]=np.sum([voc[w]*(a/(a+cnt[w]/tot)) for w in ws],axis=0)   # SUM (additive), normalised later
    return out
RP=sif_raw(P); RQ=sif_raw(Q)
u=np.linalg.svd(RP/np.maximum(np.linalg.norm(RP,axis=1,keepdims=True),1e-9),full_matrices=False)[2][0]
def sif(texts):
    Z=sif_raw(texts); Z=Z/np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-9); Z=Z-np.outer(Z@u,u); return (Z/np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-9)).astype('float32')
PB=sif(P); QB=sif(Q)
np.save('PA.npy',PA); np.save('QA.npy',QA); np.save('PB.npy',PB); np.save('QB.npy',QB)
pickle.dump(dict(items=items,tf=tf,svd=svd,u=u,cnt=cnt,tot=tot,voc={w:voc[w] for w in cnt if w in voc}),open('prep.pkl','wb'))
