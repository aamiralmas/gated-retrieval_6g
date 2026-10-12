"""E3: confidence gate as selective prediction with certified risk; shift tests; tiered (local/remote) retrieval.
E4: gate inversion under query-stuffed knowledge poisoning. Not timing-sensitive."""
import numpy as np, faiss, json, sys, re
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'.'); import enc
faiss.omp_set_num_threads(1)
SEED=20261011; items=enc.items; N=len(items)
src=np.array([i['src'] for i in items]); cat=np.array([i['cat'] for i in items]); TQ=np.where(src=='tq')[0]; OB=np.where(src=='ob')[0]
ETA=0.1
import scipy.sparse as sp
def search(P,Q,k=5,ids=None):
    if sp.issparse(P):
        PT=P.T.tocsr(); D=np.empty((Q.shape[0],k),dtype='float32'); I=np.empty((Q.shape[0],k),dtype='int64')
        for b in range(0,Q.shape[0],1000):
            S=(Q[b:b+1000]@PT).toarray(); top=np.argpartition(-S,k,axis=1)[:,:k]; sc=np.take_along_axis(S,top,1); o=np.argsort(-sc,axis=1)
            I[b:b+1000]=np.take_along_axis(top,o,1); D[b:b+1000]=np.take_along_axis(sc,o,1)
    else:
        ix=faiss.IndexFlatIP(P.shape[1]); ix.add(P); D,I=ix.search(Q,k)
    return D,(I if ids is None else ids[I])
def rowdot(A,B): return np.asarray(A.multiply(B).sum(1)).ravel() if sp.issparse(A) else (A*B).sum(1)
def vstack(A,B): return sp.vstack([A,B]).tocsr() if sp.issparse(A) else np.vstack([A,B])
def rnorm(A): return np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()) if sp.issparse(A) else np.linalg.norm(A,axis=1)
def cp_upper(k,n,eta):   # Clopper-Pearson upper bound on a binomial proportion
    return 1.0 if k>=n else float(beta.ppf(1-eta,k+1,n-k))
GRID=np.round(np.arange(1.00,0.0,-0.01),2)      # thresholds fixed in advance (m=100)
def ltt_tau(s,wrong,eps,eta=ETA):
    """Bonferroni-corrected Learn-then-Test over a fixed grid: the smallest threshold whose
    Clopper-Pearson bound at level eta/m certifies selective risk <= eps."""
    tau=np.inf; m=len(GRID)
    for t in GRID:
        a=s>=t; n=int(a.sum())
        if n and cp_upper(int(wrong[a].sum()),n,eta/m)<=eps: tau=t
    return tau
def evalgate(s,wrong,tau):
    acc=s>=tau; n=acc.sum()
    return dict(coverage=float(acc.mean()),sel_risk=float(wrong[acc].mean()) if n else 0.0,false_action=float((acc&wrong).mean()))
R={}
for E in ('A','B','C'):
    embed=enc.ENC[E][0]; raw=enc.ENC[E][1]
    if E=='C': P=embed([i['p'] for i in items]); Q=embed([i['q'] for i in items])
    else: P=np.load('P%s.npy'%E); Q=np.load('Q%s.npy'%E)
    D,I=search(P,Q); own=np.arange(N); wrong=I[:,0]!=own; s1=D[:,0]; mar=D[:,0]-D[:,1]
    w=np.exp(D/0.1); w/=w.sum(1,keepdims=True); ras=(w*D).sum(1)
    r=dict(top1_tq=float(1-wrong[TQ].mean()),top1_ob=float(1-wrong[OB].mean()),r5_tq=float((I[TQ]==own[TQ,None]).any(1).mean()),
           auroc={k:float(roc_auc_score(~wrong[TQ],v[TQ])) for k,v in (('s1',s1),('ras',ras),('margin',mar))})
    # score used by the gate = best AUROC of the three
    gname=max(r['auroc'],key=r['auroc'].get); g={'s1':s1,'ras':ras,'margin':mar}[gname]; r['gate_score']=gname
    # risk-coverage curve (TQ)
    o=np.argsort(-g[TQ]); cw=np.cumsum(wrong[TQ][o])/np.arange(1,len(o)+1); idx=np.unique(np.round(np.logspace(1,np.log10(len(o)),60)).astype(int))-1
    r['risk_cov']=dict(cov=((idx+1)/len(o)).tolist(),risk=cw[idx].tolist())
    # ---- in-distribution certification: 200 random splits ----
    rs=np.random.RandomState(SEED); ind={}
    for eps in (0.3,0.2,0.1,0.05):
        cov=[];risk=[];fa=[];inf=0
        for _ in range(200):
            p=rs.permutation(TQ); c=p[:len(p)//2]; t=p[len(p)//2:]
            tau=ltt_tau(g[c],wrong[c],eps)
            if not np.isfinite(tau): inf+=1; cov.append(0.0); risk.append(0.0); fa.append(0.0); continue
            e=evalgate(g[t],wrong[t],tau); cov.append(e['coverage']); risk.append(e['sel_risk']); fa.append(e['false_action'])
        ind[str(eps)]=dict(coverage=float(np.mean(cov)),sel_risk=float(np.mean(risk)),viol_frac=float(np.mean(np.array(risk)>eps)),false_action=float(np.mean(fa)),no_threshold_frac=inf/200)
    r['indist']=ind
    # ---- shifts: threshold certified on source, evaluated on target ----
    sh={}
    res=np.isin(cat,['Research overview','Research publications','Lexicon']); std=np.isin(cat,['Standards overview','Standards specifications'])
    for eps in (0.3,0.2):
        tau=ltt_tau(g[res],wrong[res],eps); a=evalgate(g[std],wrong[std],tau)
        tau2=ltt_tau(g[TQ],wrong[TQ],eps); b=evalgate(g[OB],wrong[OB],tau2); b0=evalgate(g[TQ],wrong[TQ],tau2)
        sh[str(eps)]=dict(research_to_standards=a,teleqna_to_oran=b,teleqna_self=b0)
    r['shift']=sh
    # ---- absent items: 30% of TeleQnA passages removed from the index ----
    rs=np.random.RandomState(SEED+1); absent=rs.choice(TQ,int(0.3*len(TQ)),replace=False); keep=np.setdiff1d(np.arange(N),absent)
    D2,I2=search(P[keep],Q,ids=keep); w2=np.exp(D2/0.1); w2/=w2.sum(1,keepdims=True)
    g2={'s1':D2[:,0],'ras':(w2*D2).sum(1),'margin':D2[:,0]-D2[:,1]}[gname]; wrong2=I2[:,0]!=own
    present=np.setdiff1d(TQ,absent); ab={}
    for eps in (0.3,0.2):
        p=rs.permutation(present); c=p[:len(p)//2]; t=p[len(p)//2:]; tau=ltt_tau(g2[c],wrong2[c],eps)
        ab[str(eps)]=dict(present=evalgate(g2[t],wrong2[t],tau),absent_accept_rate=float((g2[absent]>=tau).mean()),
                          mixed=evalgate(np.r_[g2[t],g2[absent]],np.r_[wrong2[t],np.ones(len(absent),bool)],tau))
    r['absent']=ab
    # ---- tiered retrieval: local index holds a random fraction f of the KB ----
    tier={}
    for TE in (0.2,0.3):
      for f in (0.1,0.25,0.5,0.75):
        acc=[]
        for sd in range(5):
            rs=np.random.RandomState(SEED+10+sd); loc=np.sort(rs.choice(N,int(f*N),replace=False))
            Dl,Il=search(P[loc],Q[TQ],ids=loc); wl=np.exp(Dl/0.1); wl/=wl.sum(1,keepdims=True)
            gl={'s1':Dl[:,0],'ras':(wl*Dl).sum(1),'margin':Dl[:,0]-Dl[:,1]}[gname]; wrl=Il[:,0]!=TQ
            p=rs.permutation(len(TQ)); c=p[:len(p)//2]; t=p[len(p)//2:]
            tl=ltt_tau(gl[c],wrl[c],TE); tr=ltt_tau(g[TQ][c],wrong[TQ][c],TE)
            la=gl[t]>=tl; esc=~la; ra=esc&(g[TQ][t]>=tr)
            acted=la|ra; wr=np.where(la,wrl[t],wrong[TQ][t])
            acc.append(dict(local_accept=la.mean(),local_risk=wrl[t][la].mean() if la.any() else 0.0,escalate=esc.mean(),remote_accept=ra.mean(),
                            acted=acted.mean(),e2e_risk=wr[acted].mean() if acted.any() else 0.0,fallback=(~acted).mean(),
                            remote_only_acted=(g[TQ][t]>=tr).mean(),local_top1=1-wrl[t].mean(),
                            remote_only_risk=wrong[TQ][t][g[TQ][t]>=tr].mean() if (g[TQ][t]>=tr).any() else 0.0))
        tier['%g@%g'%(f,TE)]={k:float(np.mean([a[k] for a in acc])) for k in acc[0]}
    r['tier']=tier
    # ================= E4: poisoning =================
    rs=np.random.RandomState(SEED+99); tgt=rs.choice(TQ,2000,replace=False); rest=np.setdiff1d(TQ,tgt)
    tau_clean={str(e):float(ltt_tau(g[rest],wrong[rest],e)) for e in (0.3,0.2)}
    stop=set(enc.tf.get_stop_words() or [])
    def claim(i):
        wr=[w for w in items[i]['wrong'] if len(w)>3 and 'above' not in w.lower() and 'option' not in w.lower()] or items[i]['wrong']
        return 'The correct answer is: '+wr[rs.randint(len(wr))]+'.'
    C=[claim(i) for i in tgt]
    def kw(i):
        ws=[w for w in re.findall(r"[A-Za-z0-9\-]+",items[i]['q']) if w.lower() not in stop]; rs.shuffle(ws); return ' '.join(ws)
    variants={'stuffed':[items[i]['q']+' '+c for i,c in zip(tgt,C)],'keywords':[kw(i)+'. '+c for i,c in zip(tgt,C)],'claim_only':C}
    po={}
    for vn,txt in variants.items():
        Z=embed(txt); P2=vstack(P,Z); D3,I3=search(P2,Q[tgt]); pid=N+np.arange(len(tgt))
        w3=np.exp(D3/0.1); w3/=w3.sum(1,keepdims=True); g3={'s1':D3[:,0],'ras':(w3*D3).sum(1),'margin':D3[:,0]-D3[:,1]}[gname]
        top=I3[:,0]==pid; in5=(I3==pid[:,None]).any(1); simp=rowdot(Q[tgt],Z); simown=rowdot(Q[tgt],P[tgt])
        d=dict(top1_capture=float(top.mean()),top5_capture=float(in5.mean()),sim_poison=float(simp.mean()),sim_genuine=float(simown.mean()),
               sim_clean_top1=float(s1[tgt].mean()),poison_beats_genuine=float((simp>simown).mean()))
        for e,tau in tau_clean.items():
            acc=g3>=tau; unsafe=top|(I3[:,0]!=tgt)       # acted-on evidence is not the genuine passage
            accc=g[tgt]>=tau
            d['gate_%s'%e]=dict(tau=tau,clean_coverage=float(accc.mean()),clean_risk=float(wrong[tgt][accc].mean()) if accc.any() else 0.0,
                                attack_coverage=float(acc.mean()),attack_risk=float(unsafe[acc].mean()) if acc.any() else 0.0,
                                poison_pass_given_top1=float(acc[top].mean()) if top.any() else 0.0,poison_acted=float((acc&top).mean()))
        # does the gate score separate poisoned-top1 (unsafe) from clean-correct top1 (safe)?  AUROC<0.5 => gate prefers poison
        if top.any():
            ok=~wrong[tgt]
            d['auroc_s1_safe_vs_poison']=float(roc_auc_score(np.r_[np.ones(ok.sum()),np.zeros(top.sum())],np.r_[s1[tgt][ok],D3[top,0]]))
            d['s1_poison_top1']=float(D3[top,0].mean()); d['s1_safe_top1']=float(s1[tgt][ok].mean())
        safe=g[tgt][~wrong[tgt]]; uns=g3[top]
        d['auroc_safe_vs_poison']=float(roc_auc_score(np.r_[np.ones(len(safe)),np.zeros(len(uns))],np.r_[safe,uns])) if top.any() and len(safe) else None
        if vn=='stuffed':   # closed form of Proposition (additive encoder)
            rq=raw([items[i]['q'] for i in tgt]); ri=raw(C); nq=rnorm(rq); ni=rnorm(ri); ok=(nq>1e-9)&(ni>1e-9)
            rr=ni[ok]/nq[ok]; cc=rowdot(rq,ri)[ok]/(nq[ok]*ni[ok]); pred=(1+rr*cc)/np.sqrt(1+2*rr*cc+rr**2)
            d['formula_max_abs_err']=float(np.abs(pred-simp[ok]).max()); d['r_median']=float(np.median(rr)); d['c_median']=float(np.median(cc)); d['frac_c_nonneg']=float((cc>=0).mean())
            d['lower_bound_ok']=float((simp[ok][cc>=0]>=1/np.sqrt(1+rr[cc>=0]**2)-1e-6).mean())
        po[vn]=d
    r['poison']=po; r['poison_frac_kb']=len(tgt)/(N+len(tgt))
    R[E]=r; json.dump(R,open('results/e3.json','w'),indent=1)
    print(E,'top1',r['top1_tq'],'auroc',r['auroc'],'gate',gname); print(' indist',ind); print(' shift',sh); print(' absent',ab); print(' tier',tier)
    for k,v in po.items(): print(' poison',k,{a:(round(b,3) if isinstance(b,float) else b) for a,b in v.items()})
