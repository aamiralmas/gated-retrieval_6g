"""E1: storage / build / latency / recall for standard ANN indexes. Single thread. Per-query timing."""
import numpy as np, faiss, time, json, sys
faiss.omp_set_num_threads(1); SEED=20261011; rs=np.random.RandomState(SEED)
which=sys.argv[1]
if which=='telecom':
    X=np.load('PA.npy'); Qa=np.load('QA.npy'); Qs=Qa[rs.choice(len(Qa),3000,replace=False)]; nlist=512; pq=96; Ns=[1000,3000,10000,len(X)]
else:
    G=np.load('glove400k.npy'); G=G/np.maximum(np.linalg.norm(G,axis=1,keepdims=True),1e-9); p=rs.permutation(len(G)); Qs=G[p[:3000]].copy(); X=G[p[10000:]].copy(); nlist=2048; pq=25; Ns=[10000,30000,100000,len(X)]
N,d=X.shape; K=10; raw=N*d*4
gt=faiss.IndexFlatIP(d); gt.add(X); GT=gt.search(Qs,K)[1]
def timeq(ix,Q,k=K):
    for i in range(50): ix.search(Q[i:i+1],k)
    lat=np.empty(len(Q)); I=np.empty((len(Q),k),dtype='int64')
    for i in range(len(Q)):
        t=time.perf_counter_ns(); _,ii=ix.search(Q[i:i+1],k); lat[i]=(time.perf_counter_ns()-t)/1e6; I[i]=ii[0]
    return lat,I
def rec(I,G): return float(np.mean([len(set(a)&set(b))/len(b) for a,b in zip(I,G)]))
def stat(lat): return dict(mean=float(lat.mean()),p50=float(np.percentile(lat,50)),p99=float(np.percentile(lat,99)),p999=float(np.percentile(lat,99.9)))
def size(ix): return int(faiss.serialize_index(ix).size)
out=dict(N=N,d=d,raw_bytes=raw,nq=len(Qs),indexes={})
def run(name,make,params,setp):
    t=time.time(); ix=make(); 
    ttrain=0.0
    if not ix.is_trained: ix.train(X); ttrain=time.time()-t
    t2=time.time(); ix.add(X); tadd=time.time()-t2
    r=dict(build_s=time.time()-t,train_s=ttrain,add_s=tadd,add_us_per_vec=1e6*tadd/N,bytes=size(ix),overhead=size(ix)/raw,points=[])
    for pv in params:
        setp(ix,pv); lat,I=timeq(ix,Qs); r['points'].append(dict(param=pv,recall=rec(I,GT),**stat(lat)))
    out['indexes'][name]=r; print(name,'build %.1fs size x%.2f'%(r['build_s'],r['overhead']),[(p['param'],round(p['recall'],3),round(p['p50'],3),round(p['p99'],3)) for p in r['points']],flush=True)
    json.dump(out,open('results/e1_%s.json'%which,'w'),indent=1)
    return ix
M=faiss.METRIC_INNER_PRODUCT
run('Flat',lambda:faiss.IndexFlatIP(d),[0],lambda ix,p:None)
def ivf(): q=faiss.IndexFlatIP(d); return faiss.IndexIVFFlat(q,d,nlist,M)
run('IVF-Flat',ivf,[1,2,4,8,16,32,64,128],lambda ix,p:setattr(ix,'nprobe',p))
def ivfpq(): q=faiss.IndexFlatIP(d); return faiss.IndexIVFPQ(q,d,nlist,pq,8,M)
run('IVF-PQ',ivfpq,[1,4,16,64,128],lambda ix,p:setattr(ix,'nprobe',p))
def hn(m):
    def f():
        ix=faiss.IndexHNSWFlat(d,m,M); ix.hnsw.efConstruction=200; return ix
    return f
def sete(ix,p): ix.hnsw.efSearch=p
run('HNSW16',hn(16),[16,32,64,128,256],sete)
h32=run('HNSW32',hn(32),[16,32,64,128,256],sete)
def hsq():
    ix=faiss.IndexHNSWSQ(d,faiss.ScalarQuantizer.QT_8bit,32,M); ix.hnsw.efConstruction=200; return ix
run('HNSW32-SQ8',hsq,[16,32,64,128,256],sete)
# scaling with N (Flat exact vs HNSW32, efSearch=64)
sc=[]
for n in Ns:
    Xn=X[:n]; g=faiss.IndexFlatIP(d); g.add(Xn); G2=g.search(Qs,K)[1]; lf,_=timeq(g,Qs)
    h=faiss.IndexHNSWFlat(d,32,M); h.hnsw.efConstruction=200; h.add(Xn); h.hnsw.efSearch=64; lh,I=timeq(h,Qs)
    sc.append(dict(N=n,flat=stat(lf),hnsw=stat(lh),hnsw_recall=rec(I,G2))); print('scale',n,round(sc[-1]['flat']['p50'],3),round(sc[-1]['hnsw']['p50'],3),round(sc[-1]['hnsw_recall'],3),flush=True)
out['scaling']=sc
json.dump(out,open('results/e1_%s.json'%which,'w'),indent=1)
