"""E2: per-query embedding time, corpus text size, library versions."""
import numpy as np, time, json, sys, faiss, sklearn, platform
sys.path.insert(0,'.'); import enc
rs=np.random.RandomState(20261011); it=enc.items; N=len(it)
qs=[it[i]['q'] for i in rs.choice(N,2000,replace=False)]
out={}
for nm,(f,_) in enc.ENC.items():
    for q in qs[:50]: f([q])
    lat=[]
    for q in qs:
        t=time.perf_counter_ns(); f([q]); lat.append((time.perf_counter_ns()-t)/1e6)
    lat=np.array(lat); out[nm]=dict(mean=float(lat.mean()),p50=float(np.percentile(lat,50)),p99=float(np.percentile(lat,99)))
# Encoder A through the library call (reference) and through a direct gather of the active rows (reported)
out['A_ref']=out['A']
CT=np.ascontiguousarray(enc.svd.components_.T.astype('float32'))
def fastA(q):
    x=enc.tf.transform([q]); z=x.data.astype('float32')@CT[x.indices]; return z/max(np.linalg.norm(z),1e-9)
for q in qs[:50]: fastA(q)
lat=[]
for q in qs:
    t=time.perf_counter_ns(); fastA(q); lat.append((time.perf_counter_ns()-t)/1e6)
lat=np.array(lat); out['A']=dict(mean=float(lat.mean()),p50=float(np.percentile(lat,50)),p99=float(np.percentile(lat,99)))
pb=np.array([len(i['p'].encode()) for i in it]); qb=np.array([len(i['q'].encode()) for i in it])
src=np.array([i['src'] for i in it])
out['corpus']=dict(N=N,n_tq=int((src=='tq').sum()),n_ob=int((src=='ob').sum()),text_bytes=int(pb.sum()),mean_passage_bytes=float(pb.mean()),mean_passage_bytes_tq=float(pb[src=='tq'].mean()),
                   mean_passage_bytes_ob=float(pb[src=='ob'].mean()),mean_query_bytes=float(qb.mean()),vocab=len(enc.tf.vocabulary_),
                   svd_explained=float(enc.svd.explained_variance_ratio_.sum()))
out['versions']=dict(faiss=faiss.__version__,sklearn=sklearn.__version__,numpy=np.__version__,python=platform.python_version())
json.dump(out,open('results/e2.json','w'),indent=1); print(out)
