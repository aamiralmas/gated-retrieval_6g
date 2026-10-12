"""E5: scoped local index (all passages of one category) versus the full index, for that category's queries."""
import numpy as np, json, sys
sys.path.insert(0,'.'); import enc
import importlib.util
src_code=open('e3_gate.py').read().split("R={}")[0]      # reuse search / ltt_tau / evalgate definitions
exec(src_code)
out={}
for E in ('A','B','C'):
    embed=enc.ENC[E][0]
    if E=='C': P=embed([i['p'] for i in items]); Q=embed([i['q'] for i in items])
    else: P=np.load('P%s.npy'%E); Q=np.load('Q%s.npy'%E)
    own=np.arange(N); r={}
    Df,If=search(P,Q); gf=Df[:,0]-Df[:,1]; wf=If[:,0]!=own
    for c in ['Lexicon','Research overview','Research publications','Standards overview','Standards specifications']:
        idx=np.where(cat==c)[0]; Dl,Il=search(P[idx],Q[idx],ids=idx); gl=Dl[:,0]-Dl[:,1]; wl=Il[:,0]!=idx
        res={}
        for eps in (0.3,0.2,0.1):
            cov_l=[];risk_l=[];cov_f=[];risk_f=[]
            for sd in range(50):
                rs=np.random.RandomState(1000+sd); p=rs.permutation(len(idx)); a=p[:len(p)//2]; b=p[len(p)//2:]
                tl=ltt_tau(gl[a],wl[a],eps); tf_=ltt_tau(gf[idx][a],wf[idx][a],eps)
                e1=evalgate(gl[b],wl[b],tl); e2=evalgate(gf[idx][b],wf[idx][b],tf_)
                cov_l.append(e1['coverage']); risk_l.append(e1['sel_risk']); cov_f.append(e2['coverage']); risk_f.append(e2['sel_risk'])
            res[str(eps)]=dict(cov_local=float(np.mean(cov_l)),risk_local=float(np.mean(risk_l)),cov_full=float(np.mean(cov_f)),risk_full=float(np.mean(risk_f)))
        r[c]=dict(n=len(idx),frac=len(idx)/N,top1_local=float(1-wl.mean()),top1_full=float(1-wf[idx].mean()),gate=res)
        print(E,c,r[c]['n'],round(r[c]['top1_local'],3),round(r[c]['top1_full'],3),{k:(round(v['cov_local'],3),round(v['cov_full'],3)) for k,v in res.items()},flush=True)
    out[E]=r
json.dump(out,open('results/e5.json','w'),indent=1)
