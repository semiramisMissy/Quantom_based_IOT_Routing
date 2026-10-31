"""Sample-network demo: route chosen by each protocol for one flow (first seed with a >=4-hop flow, vmax=10 m/s, t=1 s)."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import morq_sim as m
# pick the first (seed, flow) whose min-hop path has >= 4 hops, so the demo is non-trivial
for seed in range(50):
    w=m.World(seed,10); w.step()
    hit=[f for f in w.flows if len(m.dijkstra(np.where(w.adj,1.0,m.INF),*f) or [])>=5]
    if hit: s,d=hit[0]; break
rng=np.random.default_rng(1)
print(f"seed={seed}, N={m.P['N']} nodes, area {m.P['AREA']:.0f}x{m.P['AREA']:.0f} m, R={m.P['R']:.0f} m, flow {s}->{d}")
paths={}
for p in ['AODV','DSR','DijkstraC','QGA','QPSO','QACO']:
    paths[p]=m.DISC[p](w,s,d,rng,{})
q=m.MorqFlow(True); q.refresh(w,s,d)
paths['MOR-Q']=q.cands[int(np.argmax(np.sin(q.theta)**2))]
print(f"{'Protocol':10s}{'hops':>5s}{'cost':>8s}{'minLT(s)':>10s}{'P(succ)':>9s}{'E(mJ)':>8s}  path")
for p,pa in paths.items():
    a,b=pa[:-1],pa[1:]
    print(f"{p:10s}{len(pa)-1:5d}{m.path_cost(w,pa):8.3f}{m.min_lt(w,pa):10.1f}{np.prod(1-(1-w.p[a,b])**2):9.3f}{(w.E[a,b]/w.p[a,b]).sum():8.1f}  {'->'.join(map(str,pa))}")
print("MOR-Q candidate set (initial probabilities):")
for c,pr in zip(q.cands,np.sin(q.theta)**2/np.sum(np.sin(q.theta)**2)): print(f"  p={pr:.3f}  {'->'.join(map(str,c))}")
fig,ax=plt.subplots(figsize=(6.2,6.2))
I,J=np.where(np.triu(w.adj)); 
for i,j in zip(I,J): ax.plot(*zip(w.pos[i],w.pos[j]),c='#dddddd',lw=.6,zorder=1)
ax.scatter(*w.pos.T,c='#555',s=18,zorder=2)
for i,(x,y) in enumerate(w.pos): ax.text(x+8,y+8,str(i),fontsize=6)
for p,c in [('AODV','#7f7f7f'),('QGA','#1f77b4'),('MOR-Q','#d62728')]:
    pa=paths[p]; ax.plot(*w.pos[pa].T,c=c,lw=2.2,label=p,zorder=3,ls='--' if p=='AODV' else '-')
ax.scatter(*w.pos[[s,d]].T,c='k',s=90,marker='*',zorder=4); ax.legend(fontsize=8)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_aspect('equal'); fig.tight_layout(); fig.savefig('results/fig6_sample_network.png',dpi=200)
