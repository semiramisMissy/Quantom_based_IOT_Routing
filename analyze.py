"""Builds tables + figures strictly from results/raw_results.csv"""
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
df = pd.read_csv('results/raw_results.csv')
order = ['AODV','DSR','DijkstraC','QGA','QPSO','QACO','MOR-Q','MOR-Q-noMob']
metrics = {'pdr':'Packet delivery ratio','energy_per_pkt_mJ':'Energy per delivered packet (mJ)',
           'delay_ms':'Mean end-to-end delay (ms)','route_breaks':'Link-break packet drops per run','discoveries':'Route discoveries per run'}
def ci(x): return 1.96*x.std(ddof=1)/np.sqrt(len(x))
g = df.groupby(['vmax','proto'])[list(metrics)+['discoveries','ctrl_share']].agg(['mean',ci])
g.to_csv('results/summary_by_speed.csv')
# overall table (mean over all speeds & seeds)
rows=[]
for p in order:
    d=df[df.proto==p]; r={'Protocol':p}
    for m in list(metrics)+['discoveries']:
        r[m]=f"{d[m].mean():.3f} ± {ci(d[m]):.3f}" if m=='pdr' else f"{d[m].mean():.1f} ± {ci(d[m]):.1f}"
    rows.append(r)
pd.DataFrame(rows).to_csv('results/table_overall.csv',index=False)
# paired tests MOR-Q vs others (paired by seed&speed)
pt=[]
for p in order:
    if p=='MOR-Q': continue
    a=df[df.proto=='MOR-Q'].sort_values(['vmax','seed']); b=df[df.proto==p].sort_values(['vmax','seed'])
    r={'vs':p}
    for m in ['pdr','energy_per_pkt_mJ','delay_ms']:
        diff=a[m].values-b[m].values; t,pv=stats.ttest_rel(a[m].values,b[m].values)
        r[m+'_diff']=diff.mean(); r[m+'_p']=pv
    pt.append(r)
pd.DataFrame(pt).to_csv('results/paired_tests.csv',index=False)
# figures
col={'AODV':'#7f7f7f','DSR':'#bcbd22','DijkstraC':'#8c564b','QGA':'#1f77b4','QPSO':'#2ca02c','QACO':'#ff7f0e','MOR-Q':'#d62728','MOR-Q-noMob':'#e377c2'}
mk={'AODV':'o','DSR':'s','DijkstraC':'x','QGA':'^','QPSO':'v','QACO':'D','MOR-Q':'*','MOR-Q-noMob':'P'}
for i,(m,lab) in enumerate(metrics.items(),1):
    fig,ax=plt.subplots(figsize=(6.2,4))
    for p in order:
        s=df[df.proto==p].groupby('vmax')[m]
        ax.errorbar(s.mean().index,s.mean().values,yerr=s.apply(ci).values,label=p,color=col[p],marker=mk[p],
                    ms=9 if p=='MOR-Q' else 6,lw=2.4 if p=='MOR-Q' else 1.3,capsize=2,ls='--' if p=='MOR-Q-noMob' else '-')
    ax.set_xlabel('Maximum node speed (m/s)'); ax.set_ylabel(lab); ax.grid(alpha=.3)
    ax.set_xticks(sorted(df.vmax.unique()))
    if i==1: ax.legend(fontsize=7,ncol=2)
    fig.tight_layout(); fig.savefig(f'results/fig{i}_{m}.png',dpi=200); plt.close(fig)
print(pd.read_csv('results/table_overall.csv').to_string()); print(pd.read_csv('results/paired_tests.csv').round(4).to_string())
print(df.groupby(['vmax','proto'])['pdr'].mean().unstack()[order].round(3).to_string())
print(df.groupby(['vmax','proto'])['energy_per_pkt_mJ'].mean().unstack()[order].round(1).to_string())
print(df.groupby(['vmax','proto'])['delay_ms'].mean().unstack()[order].round(1).to_string())
print(df.groupby(['vmax','proto'])['discoveries'].mean().unstack()[order].round(1).to_string())
