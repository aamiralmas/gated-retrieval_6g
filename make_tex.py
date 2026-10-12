"""results/*.json -> LaTeX fragments (tables, pgfplots, number macros). No hand-typed results."""
import json, os, math, numpy as np
os.makedirs('frag',exist_ok=True)
T=json.load(open('results/e1_telecom.json')); G=json.load(open('results/e1_glove.json')); E2=json.load(open('results/e2.json')); E3=json.load(open('results/e3.json')); E5=json.load(open('results/e5.json'))
def w(n,s): open('frag/%s.tex'%n,'w').write(s)
PRE="\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.3}\n"
fi=lambda x:'{:,}'.format(int(round(x))).replace(',','{,}')
num={}
IDX=['Flat','IVF-Flat','IVF-PQ','HNSW16','HNSW32','HNSW32-SQ8']
def op(ix,target=0.95):
    ok=[p for p in ix['points'] if p['recall']>=target]
    if ok: return min(ok,key=lambda p:p['p50']),True
    return max(ix['points'],key=lambda p:p['recall']),False
# ---------------- Table: index trade-offs ----------------
def block(D,label):
    rows='\\multicolumn{8}{@{}l}{\\textit{%s}}\\\\\n'%label
    for k in IDX:
        ix=D['indexes'][k]; p,ok=op(ix)
        rows+='%s & %.2f & %.1f & %.0f & %.3f%s & %.2f & %.2f & %s \\\\\n'%(k,ix['overhead'],ix['build_s'],ix['add_us_per_vec'],p['recall'],'' if ok else '$^{\\dagger}$',p['p50'],p['p99'],'--' if k=='Flat' else str(p['param']))
    return rows
w('tab_index',"""\\begin{table*}[!t]
\\centering
\\caption{Storage, Build Cost and Query Latency of Six Index Types (Single Thread, One Query at a Time)}
\\label{tab:index}
\\footnotesize
"""+PRE+"""\\begin{threeparttable}
\\begin{tabularx}{\\textwidth}{@{}lCCCCCCC@{}}
\\toprule
\\textbf{Index} & \\textbf{Size / raw vectors} & \\textbf{Build (s)} & \\textbf{Insert ($\\mu$s/vector)} & \\textbf{Recall@10} & \\textbf{Median (ms)} & \\textbf{99th pct.\\ (ms)} & \\textbf{Setting} \\\\
\\midrule
"""+block(T,'Telecom knowledge base: $N=%s$, $d=%d$ (Encoder~A)'%(fi(T['N']),T['d']))+"\\addlinespace\n"+block(G,'GloVe word vectors: $N=%s$, $d=%d$'%(fi(G['N']),G['d']))+"""\\midrule
\\multicolumn{8}{@{}p{0.97\\textwidth}@{}}{\\footnotesize $\\dagger$ = the index never reaches recall 0.95; its best recall is shown. Otherwise the fastest setting with recall $\\ge 0.95$.}\\\\
\\bottomrule
\\end{tabularx}
\\begin{tablenotes}[flushleft]\\scriptsize
\\item Size is the serialised index divided by $4Nd$ bytes. Insert is the mean time to add one vector after any training. Setting is the number of probed lists (IVF) or the search-queue length \\emph{efSearch} (HNSW). IVF = inverted file; PQ = product quantisation; HNSW = hierarchical navigable small world; SQ8 = 8-bit scalar quantisation.
\\end{tablenotes}
\\end{threeparttable}
\\end{table*}
""")
tf=T['indexes']; gf=G['indexes']
hT,_=op(tf['HNSW32']); hG,_=op(gf['HNSW32']); h16,_=op(tf['HNSW16']); pqT,_=op(tf['IVF-PQ']); pqG,_=op(gf['IVF-PQ'])
num.update(dict(Nkb=fi(T['N']),Ntq=fi(E2['corpus']['n_tq']),Nob=fi(E2['corpus']['n_ob']),Nglove=fi(G['N']),
  FlatT='%.1f'%tf['Flat']['points'][0]['p50'],FlatTp='%.1f'%tf['Flat']['points'][0]['p99'],FlatG='%.1f'%gf['Flat']['points'][0]['p50'],FlatGp='%.1f'%gf['Flat']['points'][0]['p99'],
  HnswT='%.2f'%hT['p50'],HnswTp='%.2f'%hT['p99'],HnswTr='%.3f'%hT['recall'],HnswG='%.2f'%hG['p50'],HnswGp='%.2f'%hG['p99'],
  HnswSixT='%.2f'%h16['p50'],HnswSixTp='%.2f'%h16['p99'],
  PqSizeT='%.2f'%tf['IVF-PQ']['overhead'],PqRecT='%.2f'%pqT['recall'],PqRecG='%.2f'%pqG['recall'],PqSizeG='%.2f'%gf['IVF-PQ']['overhead'],
  SqSizeT='%.2f'%tf['HNSW32-SQ8']['overhead'],SqRecT='%.3f'%op(tf['HNSW32-SQ8'])[0]['recall'],
  GraphOvT='%.0f'%(100*(tf['HNSW32']['overhead']-1)),GraphOvG='%.0f'%(100*(gf['HNSW32']['overhead']-1)),
  InsFlatT='%.0f'%tf['Flat']['add_us_per_vec'],InsIvfT='%.0f'%tf['IVF-Flat']['add_us_per_vec'],InsHnswT='%.0f'%tf['HNSW32']['add_us_per_vec'],
  InsRatio='%.0f'%(tf['HNSW32']['add_us_per_vec']/tf['IVF-Flat']['add_us_per_vec']),
  BuildHnswT='%.1f'%tf['HNSW32']['build_s'],BuildHnswG='%.0f'%gf['HNSW32']['build_s'],
  SpeedupT='%.0f'%(tf['Flat']['points'][0]['p50']/hT['p50']),SpeedupG='%.0f'%(gf['Flat']['points'][0]['p50']/hG['p50'])))
# storage vs text
c=E2['corpus']; vecbytes=4*T['d']; num.update(dict(PassBytes='%.0f'%c['mean_passage_bytes'],PassBytesTq='%.0f'%c['mean_passage_bytes_tq'],TextMB='%.1f'%(c['text_bytes']/1e6),VecMB='%.1f'%(T['raw_bytes']/1e6),
  VecOverText='%.0f'%(T['raw_bytes']/c['text_bytes']),PqOverText='%.1f'%(tf['IVF-PQ']['bytes']/c['text_bytes']),SqOverText='%.1f'%(tf['HNSW32-SQ8']['bytes']/c['text_bytes']),
  HnswOverText='%.0f'%(tf['HNSW32']['bytes']/c['text_bytes']),FaissVer=E2['versions']['faiss'],
  EmbA='%.2f'%E2['A']['p50'],EmbAp='%.2f'%E2['A']['p99'],EmbB='%.2f'%E2['B']['p50'],EmbC='%.2f'%E2['C']['p50'],EmbAref='%.1f'%E2['A_ref']['p50']))
# ---------------- Figure: recall-latency and scaling ----------------
def coords(x,y): return ' '.join('(%.5g,%.5g)'%(a,b) for a,b in zip(x,y))
sty={'IVF-Flat':'color=blue, mark=o, thick','IVF-PQ':'color=red, mark=square, thick, dashed','HNSW32':'color=green!60!black, mark=triangle, thick','HNSW32-SQ8':'color=orange, mark=diamond, thick, dash dot','HNSW16':'color=gray, mark=x, thick, densely dotted'}
pl=''
for k in ['IVF-Flat','IVF-PQ','HNSW16','HNSW32','HNSW32-SQ8']:
    pts=tf[k]['points']; pl+='\\addplot[%s] coordinates {%s};\n\\addlegendentry{%s}\n'%(sty[k],coords([p['p50'] for p in pts],[p['recall'] for p in pts]),k)
pl+='\\addplot[only marks, mark=star, mark size=3pt, color=black] coordinates {(%.4g,1)};\n\\addlegendentry{Flat (exact)}\n'%tf['Flat']['points'][0]['p50']
figA="""\\begin{tikzpicture}
\\begin{axis}[width=0.5\\textwidth, height=6cm, xmode=log, xlabel={Median query latency (ms)}, ylabel={Recall@10},
xmin=0.04, xmax=4, ymin=0.5, ymax=1.02, grid=major, grid style={dotted,gray!40}, legend style={at={(0.5,1.04)},anchor=south,font=\\scriptsize}, legend columns=3,
, tick label style={font=\\footnotesize}, label style={font=\\small}]
"""+pl+"\\end{axis}\n\\end{tikzpicture}"
sT=T['scaling']; sG=G['scaling']
figB="""\\begin{tikzpicture}
\\begin{axis}[width=0.5\\textwidth, height=6cm, xmode=log, ymode=log, xlabel={Indexed vectors $N$}, ylabel={Median query latency (ms)},
xmin=700, xmax=600000, ymin=0.05, ymax=20, grid=major, grid style={dotted,gray!40}, legend style={at={(0.5,1.04)},anchor=south,font=\\scriptsize}, legend columns=2,
, tick label style={font=\\footnotesize}, label style={font=\\small}]
\\addplot[color=blue, mark=o, thick] coordinates {%s};
\\addlegendentry{Flat, telecom ($d=768$)}
\\addplot[color=blue, mark=triangle, thick, dashed] coordinates {%s};
\\addlegendentry{HNSW32, telecom}
\\addplot[color=red, mark=square, thick] coordinates {%s};
\\addlegendentry{Flat, GloVe ($d=100$)}
\\addplot[color=red, mark=diamond, thick, dashed] coordinates {%s};
\\addlegendentry{HNSW32, GloVe}
\\draw[gray, thin, dashed] (axis cs:700,10) -- (axis cs:600000,10);
\\node[font=\\scriptsize, anchor=north west] at (axis cs:800,10) {RT budget, 10\\,ms};
\\end{axis}
\\end{tikzpicture}"""%(coords([s['N'] for s in sT],[s['flat']['p50'] for s in sT]),coords([s['N'] for s in sT],[s['hnsw']['p50'] for s in sT]),coords([s['N'] for s in sG],[s['flat']['p50'] for s in sG]),coords([s['N'] for s in sG],[s['hnsw']['p50'] for s in sG]))
w('fig_index',"""\\begin{figure*}[!t]
\\centering
\\resizebox{0.86\\textwidth}{!}{%
"""+figA+"\\hspace{4mm}"+figB+"""%
}
\\caption{Left: recall against median single-query latency on the telecom knowledge base as each index's search parameter is varied. Right: median latency against index size for exact search and for HNSW32 with \\emph{efSearch}\\,$=64$; exact search grows linearly and approaches the 10\\,ms real-time budget near $4\\times10^{5}$ vectors of dimension 100, while graph search stays below 0.3\\,ms throughout.}
\\label{fig:index}
\\end{figure*}
""")
num['FlatCross']='$%.0f\\times10^{4}$'%(10/tf['Flat']['points'][0]['p50']*T['N']/1e4)
num.update(dict(ScaleFlatBig='%.1f'%sG[-1]['flat']['p50'],ScaleHnswBig='%.2f'%sG[-1]['hnsw']['p50'],ScaleHnswRecBig='%.2f'%sG[-1]['hnsw_recall']))
# ---------------- Table: latency budget ----------------
emb=E2['A']['p50']; ann=hT['p50']; flat=tf['Flat']['points'][0]['p50']; GEN=[('none',0.0),('117$^{a}$',117.0),('499$^{b}$',499.0)]
rows=''
for tier,B in (('Real-time',10.0),('Near-RT',1000.0)):
    for i,(gl,g) in enumerate(GEN):
        H=B-g-emb; feas=H>ann; net=H-ann
        rows+='%s & %s & %s & %s & %s & %s \\\\\n'%(tier if i==0 else '', '%g'%B if i==0 else '', gl, '%.2f'%H if H>0 else '$<0$', 'yes' if feas else 'no', '%.1f'%net if feas else '--')
    rows+='\\addlinespace\n'
w('tab_budget',"""\\begin{table}[!t]
\\centering
\\caption{Retrieval Headroom by Control Tier}
\\label{tab:budget}
\\footnotesize
"""+PRE+"""\\begin{threeparttable}
\\begin{tabularx}{\\columnwidth}{@{}lCCCCC@{}}
\\toprule
\\textbf{Tier} & \\textbf{$B_j$} & \\textbf{$T_{\\mathrm{gen}}$} & \\textbf{$H_j(\\loc)$} & \\textbf{Feasible} & \\textbf{Max.\\ $T_{\\mathrm{net}}$} \\\\
\\midrule
"""+rows+"""\\midrule
\\multicolumn{6}{@{}p{0.97\\columnwidth}@{}}{\\footnotesize All times in ms. Measured: $T_{\\mathrm{emb}}=%.2f$, $T_{\\mathrm{ann}}=%.2f$ (HNSW32); $T_{\\mathrm{val}}$ taken as zero.}\\\\
\\bottomrule
\\end{tabularx}
\\begin{tablenotes}[flushleft]\\scriptsize
\\item $T_{\\mathrm{gen}}$ is cited, not measured: single-query warm latency reported in \\cite{6gneedsagents2026} on a data-centre GPU for $^{a}$a quantised 135-million-parameter model and $^{b}$an 8-billion-parameter model; ``none'' is retrieval without generation. Max.\\ $T_{\\mathrm{net}}$ is the largest round trip for which a remote index is still feasible, by Proposition~\\ref{prop:feasible}(ii).
\\end{tablenotes}
\\end{threeparttable}
\\end{table}
"""%(emb,ann))
num.update(dict(HeadRT='%.1f'%(10-emb-ann),HeadRTflat='%.1f'%(10-emb-flat),NetNrtSmall='%.0f'%(1000-117-emb-ann),NetNrtBig='%.0f'%(1000-499-emb-ann),RetShareNrt='%.2f'%(100*(emb+ann)/1000),
  TokRtx='%.1f'%(1000/131.7),TokHailo='%.0f'%(1000/6.9)))
# ---------------- Table: gate certification ----------------
EN={'A':'A (LSA, $d=768$)','B':'B (GloVe-SIF, $d=100$)','C':'C (sparse lexical)'}
rows=''
for e in 'ABC':
    r=E3[e]; ind=r['indist']
    cell=lambda eps:('%.1f / %.1f'%(100*ind[eps]['coverage'],100*ind[eps]['sel_risk'])) if ind[eps]['no_threshold_frac']<0.5 else 'none'
    rows+='%s & %.1f & %.2f & %.2f & %.2f & %s & %s & %s \\\\\n'%(EN[e],100*r['top1_tq'],r['auroc']['s1'],r['auroc']['ras'],r['auroc']['margin'],cell('0.3'),cell('0.2'),cell('0.1'))
w('tab_gate',"""\\begin{table*}[!t]
\\centering
\\caption{Gate Scores and Certified Operating Points on TeleQnA Queries (200 Random Calibration/Test Splits, $\\eta=0.1$)}
\\label{tab:gate}
\\footnotesize
"""+PRE+"""\\begin{threeparttable}
\\begin{tabularx}{\\textwidth}{@{}p{3.6cm}CCCCCCC@{}}
\\toprule
 & & \\multicolumn{3}{c}{\\textbf{AUROC of gate score}} & \\multicolumn{3}{c}{\\textbf{Coverage / realised risk (\\%) at target $\\varepsilon$}} \\\\
\\cmidrule(lr){3-5}\\cmidrule(l){6-8}
\\textbf{Encoder} & \\textbf{Top-1 (\\%)} & \\textbf{Top-1 sim.} & \\textbf{Weighted top-5} & \\textbf{Margin} & \\textbf{$\\varepsilon=0.3$} & \\textbf{$\\varepsilon=0.2$} & \\textbf{$\\varepsilon=0.1$} \\\\
\\midrule
"""+rows+"""\\midrule
\\multicolumn{8}{@{}p{0.97\\textwidth}@{}}{\\footnotesize Coverage = share of test queries the gate lets the agent act on; risk = share of those whose evidence is wrong; none = no threshold could be certified.}\\\\
\\bottomrule
\\end{tabularx}
\\begin{tablenotes}[flushleft]\\scriptsize
\\item AUROC = area under the ROC curve for separating correct from wrong top-1 retrievals. Weighted top-5 is the softmax-weighted mean similarity of the five best passages (temperature 0.1). The margin is used as the gate score in all later experiments. Entries are means over splits.
\\end{tablenotes}
\\end{threeparttable}
\\end{table*}
""")
viol=max(E3[e]['indist'][k]['viol_frac'] for e in 'ABC' for k in ('0.3','0.2','0.1'))
num.update(dict(TopA='%.1f\\%%'%(100*E3['A']['top1_tq']),TopB='%.1f\\%%'%(100*E3['B']['top1_tq']),TopC='%.1f\\%%'%(100*E3['C']['top1_tq']),TopOb='%.1f'%(100*max(E3[e]['top1_ob'] for e in 'ABC')),
  AucMarLo='%.2f'%min(E3[e]['auroc']['margin'] for e in 'ABC'),AucMarHi='%.2f'%max(E3[e]['auroc']['margin'] for e in 'ABC'),
  AucSimLo='%.2f'%min(E3[e]['auroc']['s1'] for e in 'ABC'),AucSimHi='%.2f'%max(E3[e]['auroc']['s1'] for e in 'ABC'),
  AucRasLo='%.2f'%min(E3[e]['auroc']['ras'] for e in 'ABC'),AucRasHi='%.2f'%max(E3[e]['auroc']['ras'] for e in 'ABC'),
  ViolMax='%.1f'%(100*viol),CovCtwo='%.1f'%(100*E3['C']['indist']['0.2']['coverage']),RiskCtwo='%.1f'%(100*E3['C']['indist']['0.2']['sel_risk']),
  CovCthree='%.1f'%(100*E3['C']['indist']['0.3']['coverage']),CovAtwo='%.1f'%(100*E3['A']['indist']['0.2']['coverage']),CovBtwo='%.1f'%(100*E3['B']['indist']['0.2']['coverage']),
  NoThrC='%.0f'%(100*E3['C']['indist']['0.1']['no_threshold_frac'])))
# ---------------- Table: shift ----------------
rows=''
for e in 'ABC':
    r=E3[e]; s=r['shift']['0.2']; a=r['absent']['0.2']
    f=lambda d:'%.1f / %.1f'%(100*d['coverage'],100*d['sel_risk'])
    rows+='%s & %s & %s & %s & %s & %.1f \\\\\n'%(EN[e],f(s['teleqna_self']),f(s['research_to_standards']),f(s['teleqna_to_oran']),f(a['mixed']),100*a['absent_accept_rate'])
w('tab_shift',"""\\begin{table*}[!t]
\\centering
\\caption{A Threshold Certified at $\\varepsilon=0.2$ Used Outside Its Calibration Distribution: Coverage / Realised Risk (\\%)}
\\label{tab:shift}
\\footnotesize
"""+PRE+"""\\begin{threeparttable}
\\begin{tabularx}{\\textwidth}{@{}p{3.6cm}CCCCC@{}}
\\toprule
\\textbf{Encoder} & \\textbf{Same distribution} & \\textbf{Research $\\rightarrow$ standards} & \\textbf{TeleQnA $\\rightarrow$ ORAN-Bench} & \\textbf{30\\% of passages deleted} & \\textbf{Accepted when passage absent (\\%)} \\\\
\\midrule
"""+rows+"""\\midrule
\\multicolumn{6}{@{}p{0.97\\textwidth}@{}}{\\footnotesize The certificate promises risk $\\le 20\\%$ only in the first column; every risk above 20 in the other columns is a silent violation.}\\\\
\\bottomrule
\\end{tabularx}
\\begin{tablenotes}[flushleft]\\scriptsize
\\item Same distribution: calibrated and evaluated on all TeleQnA queries (in-sample reference). Research $\\rightarrow$ standards: calibrated on lexicon and research questions, evaluated on standards questions. Deleted: threshold calibrated on queries whose passage is present, evaluated on a mix of present and absent queries; the last column is the share of absent-passage queries the gate accepts.
\\end{tablenotes}
\\end{threeparttable}
\\end{table*}
""")
num.update(dict(ShiftObLo='%.0f'%(100*min(E3[e]['shift']['0.2']['teleqna_to_oran']['sel_risk'] for e in 'ABC')),ShiftObHi='%.0f'%(100*max(E3[e]['shift']['0.2']['teleqna_to_oran']['sel_risk'] for e in 'ABC')),
  ShiftStdC='%.0f'%(100*E3['C']['shift']['0.2']['research_to_standards']['sel_risk']),ShiftStdA='%.0f'%(100*E3['A']['shift']['0.2']['research_to_standards']['sel_risk']),ShiftStdB='%.0f'%(100*E3['B']['shift']['0.2']['research_to_standards']['sel_risk']),
  AbsLo='%.0f'%(100*min(E3[e]['absent']['0.2']['mixed']['sel_risk'] for e in 'ABC')),AbsHi='%.0f'%(100*max(E3[e]['absent']['0.2']['mixed']['sel_risk'] for e in 'ABC'))))
# ---------------- Table: scoped local index ----------------
rows=''
for e in 'BC':
    rows+='\\multicolumn{7}{@{}l}{\\textit{Encoder %s}}\\\\\n'%EN[e]
    for c,v in E5[e].items():
        g=v['gate']; rows+='%s & %s & %.1f & %.1f / %.1f & %.1f / %.1f & %.1f / %.1f & %.1f / %.1f \\\\\n'%(c,fi(v['n']),100*v['frac'],100*v['top1_local'],100*v['top1_full'],100*g['0.3']['cov_local'],100*g['0.3']['cov_full'],100*g['0.2']['cov_local'],100*g['0.2']['cov_full'],100*g['0.1']['cov_local'],100*g['0.1']['cov_full'])
    rows+='\\addlinespace\n'
w('tab_scoped',"""\\begin{table*}[!t]
\\centering
\\caption{A Local Index Scoped to One Query Class Against the Full Index, for That Class's Queries (Scoped / Full)}
\\label{tab:scoped}
\\footnotesize
"""+PRE+"""\\begin{threeparttable}
\\begin{tabularx}{\\textwidth}{@{}p{3.6cm}CCCCCC@{}}
\\toprule
 & & & & \\multicolumn{3}{c}{\\textbf{Certified coverage (\\%) at target $\\varepsilon$}} \\\\
\\cmidrule(l){5-7}
\\textbf{Query class} & \\textbf{$N_{\\loc}$} & \\textbf{Share of KB (\\%)} & \\textbf{Top-1 (\\%)} & \\textbf{$\\varepsilon=0.3$} & \\textbf{$\\varepsilon=0.2$} & \\textbf{$\\varepsilon=0.1$} \\\\
\\midrule
"""+rows.replace(' & '.join(['%s']*1),'%s')+"""\\midrule
\\multicolumn{7}{@{}p{0.97\\textwidth}@{}}{\\footnotesize Each cell gives scoped local index / full index. A coverage of 0.0 means no threshold could be certified.}\\\\
\\bottomrule
\\end{tabularx}
\\begin{tablenotes}[flushleft]\\scriptsize
\\item The scoped index holds all passages of one TeleQnA category and is queried only with that category's questions; the full index holds the whole knowledge base. Thresholds are certified per class with $\\eta=0.1$; means over 50 random calibration/test splits. KB = knowledge base.
\\end{tablenotes}
\\end{threeparttable}
\\end{table*}
""".replace('{@{}p{3.6cm}CCCCCC@{}}','{@{}p{3.6cm}CCCCCC@{}}'))
ss=E5['C']['Standards specifications']; tr=E3['C']['tier']
num.update(dict(ScSpecLoc='%.1f'%(100*ss['top1_local']),ScSpecFull='%.1f'%(100*ss['top1_full']),ScSpecCovLoc='%.1f'%(100*ss['gate']['0.1']['cov_local']),ScSpecCovFull='%.1f'%(100*ss['gate']['0.1']['cov_full']),ScSpecShare='%.1f'%(100*ss['frac']),
  TierHalfAcc='%.1f'%(100*tr['0.5@0.2']['local_accept']),TierThreeQAcc='%.1f'%(100*tr['0.75@0.2']['local_accept']),TierThreeQTop='%.1f'%(100*tr['0.75@0.2']['local_top1']),
  TierThreeQActed='%.1f'%(100*tr['0.75@0.2']['acted']),TierRemoteActed='%.1f'%(100*tr['0.75@0.2']['remote_only_acted']),TierThreeQRisk='%.1f'%(100*tr['0.75@0.2']['e2e_risk']),TierRemoteRisk='%.1f'%(100*tr['0.75@0.2']['remote_only_risk']),
  TierMaxAccAB='%.1f'%(100*max(E3[e]['tier']['%s@0.2'%f]['local_accept'] for e in 'AB' for f in ('0.1','0.25','0.5','0.75')))))
# ---------------- Table: poisoning ----------------
VN={'stuffed':'Query + payload','keywords':'Keywords + payload','claim_only':'Payload only'}
rows=''
for e in 'ABC':
    rows+='\\multicolumn{8}{@{}l}{\\textit{Encoder %s}}\\\\\n'%EN[e]
    for v in ('stuffed','keywords','claim_only'):
        d=E3[e]['poison'][v]; g=d['gate_0.2']
        rows+='%s & %.1f & %.1f & %.2f / %.2f & %s & %.1f / %.1f & %.1f / %.1f & %.1f \\\\\n'%(VN[v],100*d['top1_capture'],100*d['top5_capture'],d['sim_poison'],d['sim_genuine'],('%.2f'%d['auroc_s1_safe_vs_poison']) if d.get('auroc_s1_safe_vs_poison') is not None else '--',
            100*g['clean_coverage'],100*g['clean_risk'],100*g['attack_coverage'],100*g['attack_risk'],100*g['poison_acted'])
    rows+='\\addlinespace\n'
w('tab_poison',"""\\begin{table*}[!t]
\\centering
\\caption{One Injected Passage per Target Query (2{,}000 Targets): Retrieval Capture and Behaviour of a Gate Certified at $\\varepsilon=0.2$}
\\label{tab:poison}
\\footnotesize
"""+PRE+"""\\begin{threeparttable}
\\begin{tabularx}{\\textwidth}{@{}p{2.9cm}CCCCCCC@{}}
\\toprule
 & \\multicolumn{2}{c}{\\textbf{Injected passage ranked (\\%)}} & \\textbf{Similarity} & \\textbf{AUROC} & \\multicolumn{2}{c}{\\textbf{Gate: coverage / risk (\\%)}} & \\textbf{Acted on} \\\\
\\cmidrule(lr){2-3}\\cmidrule(lr){6-7}
\\textbf{Attacker knowledge} & \\textbf{first} & \\textbf{in top 5} & \\textbf{inj.\\ / genuine} & \\textbf{of sim.} & \\textbf{no attack} & \\textbf{under attack} & \\textbf{injection (\\%)} \\\\
\\midrule
"""+rows+"""\\midrule
\\multicolumn{8}{@{}p{0.97\\textwidth}@{}}{\\footnotesize AUROC of top-1 similarity for separating correct clean retrievals (positive) from retrievals captured by the injection; below 0.5 the score prefers the injection.}\\\\
\\bottomrule
\\end{tabularx}
\\begin{tablenotes}[flushleft]\\scriptsize
\\item Payload: a sentence asserting one of the question's own incorrect options. Similarity: mean cosine between the query and the injected passage, and between the query and its genuine passage. Risk under attack counts any acted-on evidence other than the genuine passage. The attack is simulated on real questions and distractors; the injected passages are %.1f\\%% of the knowledge base.
\\end{tablenotes}
\\end{threeparttable}
\\end{table*}
"""%(100*E3['A']['poison_frac_kb']))
P=lambda e,v:E3[e]['poison'][v]
num.update(dict(CapLo='%.0f'%(100*min(P(e,'stuffed')['top1_capture'] for e in 'ABC')),CapHi='%.0f'%(100*max(P(e,'stuffed')['top1_capture'] for e in 'ABC')),
  CapFiveLo='%.0f'%(100*min(P(e,'stuffed')['top5_capture'] for e in 'ABC')),CapFiveHi='%.0f'%(100*max(P(e,'stuffed')['top5_capture'] for e in 'ABC')),
  AtkRiskLo='%.0f'%(100*min(P(e,'stuffed')['gate_0.2']['attack_risk'] for e in 'ABC')),CleanRiskLo='%.0f'%(100*min(P(e,'stuffed')['gate_0.2']['clean_risk'] for e in 'ABC')),CleanRiskHi='%.0f'%(100*max(P(e,'stuffed')['gate_0.2']['clean_risk'] for e in 'ABC')),
  AucInvC='%.2f'%P('C','stuffed')['auroc_s1_safe_vs_poison'],AucInvA='%.2f'%P('A','stuffed')['auroc_s1_safe_vs_poison'],AucInvB='%.2f'%P('B','stuffed')['auroc_s1_safe_vs_poison'],
  CovCleanC='%.1f'%(100*P('C','stuffed')['gate_0.2']['clean_coverage']),CovAtkC='%.1f'%(100*P('C','stuffed')['gate_0.2']['attack_coverage']),
  CovCleanA='%.1f'%(100*P('A','stuffed')['gate_0.2']['clean_coverage']),CovAtkA='%.1f'%(100*P('A','stuffed')['gate_0.2']['attack_coverage']),
  ClaimCapHi='%.1f'%(100*max(P(e,'claim_only')['top1_capture'] for e in 'ABC')),FormErrAB='%.0e'%max(P(e,'stuffed')['formula_max_abs_err'] for e in 'AB'),FormErrC='%.3f'%P('C','stuffed')['formula_max_abs_err'],
  Rmed='%.2f'%P('A','stuffed')['r_median'],RmedB='%.2f'%P('B','stuffed')['r_median'],CnonnegLo='%.0f'%(100*min(P(e,'stuffed')['frac_c_nonneg'] for e in 'ABC')),
  LbOk='%.0f'%(100*min(P(e,'stuffed')['lower_bound_ok'] for e in 'ABC')),PoisonFrac='%.1f'%(100*E3['A']['poison_frac_kb']),
  SimInjA='%.2f'%P('A','stuffed')['sim_poison'],SimGenA='%.2f'%P('A','stuffed')['sim_genuine'],KwCapC='%.0f'%(100*P('C','keywords')['top1_capture'])))
# ---------------- Figure: risk-coverage and scoped bars ----------------
pl=''
st={'A':'color=blue, thick','B':'color=red, thick, dashed','C':'color=green!60!black, thick, dash dot'}
for e in 'ABC':
    rc=E3[e]['risk_cov']; pl+='\\addplot[%s, mark=none] coordinates {%s};\n\\addlegendentry{Encoder %s}\n'%(st[e],coords(rc['cov'],rc['risk']),e)
figC="""\\begin{tikzpicture}
\\begin{axis}[width=0.5\\textwidth, height=6cm, xmode=log, xlabel={Coverage (share of queries acted on)}, ylabel={Selective risk},
xmin=0.001, xmax=1, ymin=0, ymax=0.75, clip=false, grid=major, grid style={dotted,gray!40}, legend style={at={(0.5,1.04)},anchor=south,font=\\scriptsize}, legend columns=3,
, tick label style={font=\\footnotesize}, label style={font=\\small}]
"""+pl+"""\\draw[gray, thin, dashed] (axis cs:0.001,0.2) -- (axis cs:1,0.2);
\\node[font=\\scriptsize, anchor=west] at (axis cs:1,0.2) {$\\varepsilon{=}0.2$};
\\draw[gray, thin, dashed] (axis cs:0.001,0.1) -- (axis cs:1,0.1);
\\node[font=\\scriptsize, anchor=west] at (axis cs:1,0.1) {$\\varepsilon{=}0.1$};
\\end{axis}
\\end{tikzpicture}"""
cats=list(E5['C']); lab=['Lexicon','Res.\\ overview','Res.\\ publ.','Std.\\ overview','Std.\\ specs']
figD="""\\begin{tikzpicture}
\\begin{axis}[width=0.5\\textwidth, height=6cm, ybar, bar width=7pt, ymin=0, ymax=100, ylabel={Certified coverage at $\\varepsilon=0.2$ (\\%%)},
symbolic x coords={%s}, xtick=data, x tick label style={font=\\scriptsize, rotate=20, anchor=east}, grid=major, grid style={dotted,gray!40},
legend style={at={(0.5,1.04)},anchor=south,font=\\scriptsize}, legend columns=2, tick label style={font=\\footnotesize}, label style={font=\\small}, enlarge x limits=0.15]
\\addplot[fill=blue!55, draw=blue!80!black] coordinates {%s};
\\addlegendentry{Scoped local index}
\\addplot[fill=orange!60, draw=orange!80!black, postaction={pattern=north east lines}] coordinates {%s};
\\addlegendentry{Full index}
\\end{axis}
\\end{tikzpicture}"""%(','.join(lab),' '.join('(%s,%.1f)'%(l,100*E5['C'][c]['gate']['0.2']['cov_local']) for l,c in zip(lab,cats)),' '.join('(%s,%.1f)'%(l,100*E5['C'][c]['gate']['0.2']['cov_full']) for l,c in zip(lab,cats)))
w('fig_gate',"""\\begin{figure*}[!t]
\\centering
\\resizebox{0.86\\textwidth}{!}{%
"""+figC+"\\hspace{4mm}"+figD+"""%
}
\\caption{Left: selective risk against coverage on TeleQnA queries when the margin gate is tightened; a certificate at target $\\varepsilon$ is possible only where a curve lies below the corresponding line with enough accepted queries (Corollary~\\ref{cor:burden}). Right: share of a class's queries on which the agent may act with risk certified at $\\varepsilon=0.2$, for a local index scoped to that class and for the full index (Encoder~C).}
\\label{fig:gate}
\\end{figure*}
""")
# ---------------- Table: freshness ----------------
rows=''
LAM=1/60.0; PHI=0.1
for k in ['Flat','IVF-Flat','HNSW32','HNSW32-SQ8']:
    ix=tf[k]; u=ix['add_us_per_vec']*1e-6; rb=ix['build_s']; x=LAM*max(rb,1e-9)
    floor=1-math.exp(-x)*(1-math.exp(-x))/x
    rows+='%s & %.0f & %.1f & %s & %s & %.1f \\\\\n'%(k,ix['add_us_per_vec'],rb,fi(PHI/(1.0*u)),fi(PHI/(LAM*u)),100*floor)
w('tab_fresh',"""\\begin{table}[!t]
\\centering
\\caption{Freshness Capacity of a Local Index (Telecom Set, $d=768$)}
\\label{tab:fresh}
\\footnotesize
"""+PRE+"""\\begin{threeparttable}
\\begin{tabularx}{\\columnwidth}{@{}lCCCCC@{}}
\\toprule
 & & & \\multicolumn{2}{c}{\\textbf{$N_{\\max}$ for $\\lambda=$}} & \\\\
\\cmidrule(lr){4-5}
\\textbf{Index} & \\textbf{$u$ ($\\mu$s)} & \\textbf{$\\rho_b$ (s)} & \\textbf{1/s} & \\textbf{1/min} & \\textbf{Floor (\\%)} \\\\
\\midrule
"""+rows+"""\\midrule
\\multicolumn{6}{@{}p{0.97\\columnwidth}@{}}{\\footnotesize $N_{\\max}$ at $\\phi=0.1$. Floor = lowest $P_{\\mathrm{stale}}$ under continuous full rebuilds, $\\lambda=1$/min.}\\\\
\\bottomrule
\\end{tabularx}
\\begin{tablenotes}[flushleft]\\scriptsize
\\item $u$ and $\\rho_b$ are measured (Table~\\ref{tab:index}); $N_{\\max}$ follows \\eqref{eq:capacity} and the floor follows \\eqref{eq:stale} with $\\Delta=\\rho_b$. Embedding cost is excluded; with a neural encoder it adds to $u$. $\\lambda$ and $\\phi$ are illustrative parameters, not measurements.
\\end{tablenotes}
\\end{threeparttable}
\\end{table}
""")
uH=tf['HNSW32']['add_us_per_vec']*1e-6; uI=tf['IVF-Flat']['add_us_per_vec']*1e-6
num.update(dict(CapHnswSec=fi(PHI/uH),CapIvfSec=fi(PHI/uI),CapHnswMin=fi(PHI/(LAM*uH))))
w('numbers',''.join('\\newcommand{\\N%s}{%s}\n'%(k,v) for k,v in num.items()))
for k,v in num.items(): print(k,v,end=' | ')
