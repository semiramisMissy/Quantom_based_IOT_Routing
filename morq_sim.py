"""
MOR-Q: Mobility-aware Optimized Routing with Quantum-inspired Probability Updates
Lightweight time-stepped simulator (NOT NS-3 / OMNeT++).

Protocols: AODV*, DSR*, DijkstraC (classical composite baseline), QGA, QPSO, QACO,
           MOR-Q, MOR-Q-noMob (ablation).   (* = simplified re-implementations)
Run:  python morq_sim.py            -> results/raw_results.csv
"""
import heapq, math, sys, os, time
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ parameters
P = dict(N=40, AREA=1000.0, R=300.0, T=150, N_FLOWS=8, VMIN=0.5,
         E_ELEC=2.0, E_AMP=4e-5,       # mJ per hop (tx+rx), mJ/m^2
         E_FLOOD=2.0,                  # mJ per node per route-discovery flood
         T_LT=30.0,                    # lifetime cap (s)
         HOP_MS=5.0, RETRY_MS=10.0, DISC_MS=100.0,
         W=(0.30, 0.20, 0.30, 0.20),   # energy, delay, stability, reliability
         TAU_MIN=4.0, K_CAND=4, LAMBDA=1.5,
         POP=8, ITER=8,
         BEACON_HELLO=0.05, BEACON_MOB=0.20)   # mJ per node per second
INF = 1e9

# ------------------------------------------------------------------ mobility / network state
class World:
    def __init__(self, seed, vmax):
        r = np.random.default_rng(seed)
        N, A = P['N'], P['AREA']
        self.rng = r
        self.vmax = vmax
        self.pos = r.uniform(0, A, (N, 2))
        self.tgt = r.uniform(0, A, (N, 2))
        self.spd = r.uniform(P['VMIN'], vmax, N)
        self.vel = np.zeros((N, 2))
        self._upd_vel()
        pairs = []
        while len(pairs) < P['N_FLOWS']:
            s, d = r.choice(N, 2, replace=False)
            if (s, d) not in pairs: pairs.append((int(s), int(d)))
        self.flows = pairs
        self.compute()

    def _upd_vel(self):
        dv = self.tgt - self.pos
        n = np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9
        self.vel = dv / n * self.spd[:, None]

    def step(self, dt=1.0):
        dv = self.tgt - self.pos
        dist = np.linalg.norm(dv, axis=1)
        arrive = dist <= self.spd * dt
        self.pos = np.where(arrive[:, None], self.tgt, self.pos + self.vel * dt)
        k = int(arrive.sum())
        if k:
            self.tgt[arrive] = self.rng.uniform(0, P['AREA'], (k, 2))
            self.spd[arrive] = self.rng.uniform(P['VMIN'], self.vmax, k)
        self._upd_vel()
        self.compute()

    def compute(self):
        R, N = P['R'], P['N']
        dp = self.pos[:, None, :] - self.pos[None, :, :]
        D = np.linalg.norm(dp, axis=2)
        self.D = D
        self.adj = (D < R) & ~np.eye(N, dtype=bool)
        self.p = np.where(self.adj, np.maximum(0.05, 0.99 * (1 - (D / R) ** 3)), 0.0)
        # predicted link lifetime: solve |dp + dv t| = R
        dv = self.vel[:, None, :] - self.vel[None, :, :]
        a = (dv ** 2).sum(2) + 1e-12
        b = (dp * dv).sum(2)
        c = (dp ** 2).sum(2) - R ** 2
        disc = np.maximum(b * b - a * c, 0)
        t = (-b + np.sqrt(disc)) / a
        self.LT = np.where(self.adj, np.clip(t, 0, P['T_LT']), 0.0)
        self.E = np.where(self.adj, P['E_ELEC'] + P['E_AMP'] * D ** 2, 0.0)
        # additive edge cost used for path search (composite surrogate)
        We, Wd, Ws, Wf = P['W']
        with np.errstate(divide='ignore', invalid='ignore'):
            c_ = (We * (self.E / self.p) / 4.5 + Wd * (1 / self.p) +
                  Ws * (1 - self.LT / P['T_LT']) + Wf * (1 - self.p))
        self.C = np.where(self.adj, c_, INF)
        # variant without stability term (used by ablation)
        with np.errstate(divide='ignore', invalid='ignore'):
            c2 = (We * (self.E / self.p) / 4.5 + Wd * (1 / self.p) + Wf * (1 - self.p))
        self.C_nomob = np.where(self.adj, c2 + Ws * 0.5, INF)

# ------------------------------------------------------------------ path utilities
def dijkstra(C, s, d):
    N = C.shape[0]
    dist = np.full(N, INF); prev = -np.ones(N, int); dist[s] = 0
    done = np.zeros(N, bool)
    for _ in range(N):
        u = int(np.argmin(np.where(done, INF * 2, dist)))
        if dist[u] >= INF or u == d: break
        done[u] = True
        nd = dist[u] + C[u]
        m = (nd < dist) & ~done
        dist[m] = nd[m]; prev[m] = u
    if dist[d] >= INF: return None
    path = [d]
    while path[-1] != s: path.append(int(prev[path[-1]]))
    return path[::-1]

def path_cost(w, path):
    """Non-additive path-level cost (bottleneck lifetime, product reliability)."""
    if path is None: return INF
    a, b = path[:-1], path[1:]
    We, Wd, Ws, Wf = P['W']
    e = (w.E[a, b] / w.p[a, b]).sum()
    dly = (1 / w.p[a, b]).sum()
    lt = w.LT[a, b].min()
    rel = np.prod(1 - (1 - w.p[a, b]) ** 2)
    return (We * e / (4.5 * 6) + Wd * dly / 6 + Ws * (1 - lt / P['T_LT']) + Wf * (1 - rel))

def valid(w, path):
    return path is not None and all(w.adj[a, b] for a, b in zip(path[:-1], path[1:]))

def min_lt(w, path):
    return min(w.LT[a, b] for a, b in zip(path[:-1], path[1:]))

def k_diverse(w, s, d, K, C):
    C2 = C.copy(); out = []
    for _ in range(K):
        p = dijkstra(C2, s, d)
        if p is None: break
        if p not in out: out.append(p)
        for a, b in zip(p[:-1], p[1:]):
            C2[a, b] *= 1.8; C2[b, a] *= 1.8
    return out

def decode(w, s, d, wt):
    """Node-weight vector -> path via Dijkstra on perturbed cost."""
    Cp = w.C * (1 + P['LAMBDA'] * (wt[:, None] + wt[None, :]) / 2)
    return dijkstra(Cp, s, d)

# ------------------------------------------------------------------ route discovery strategies
def disc_aodv(w, s, d, rng, st):
    return dijkstra(np.where(w.adj, 1.0, INF), s, d)

def disc_dsr(w, s, d, rng, st):
    ps = k_diverse(w, s, d, 3, np.where(w.adj, 1.0, INF))
    st['cache'] = ps[1:]
    return ps[0] if ps else None

def disc_dijkstraC(w, s, d, rng, st):
    return dijkstra(w.C, s, d)

def disc_qga(w, s, d, rng, st):
    N = P['N']; th = np.full(N, math.pi / 4)
    best, bc = dijkstra(w.C, s, d), INF
    if best is None: return None
    bc = path_cost(w, best); gbits = None
    for _ in range(P['ITER']):
        for _ in range(P['POP']):
            bits = (rng.random(N) < np.sin(th) ** 2).astype(float)
            p = decode(w, s, d, bits); c = path_cost(w, p)
            if c < bc: best, bc, gbits = p, c, bits
        if gbits is not None:  # rotation gate toward best chromosome
            th += np.where(gbits > 0.5, 1, -1) * 0.06 * math.pi
            th = np.clip(th, 0.05, math.pi / 2 - 0.05)
    return best

def disc_qpso(w, s, d, rng, st):
    N = P['N']; M = P['POP']
    x = rng.random((M, N)); pb = x.copy(); pc = np.full(M, INF)
    best, bc, g = dijkstra(w.C, s, d), INF, x[0].copy()
    if best is None: return None
    bc = path_cost(w, best)
    for it in range(P['ITER']):
        for i in range(M):
            c = path_cost(w, decode(w, s, d, x[i]))
            if c < pc[i]: pc[i], pb[i] = c, x[i].copy()
            if c < bc: bc, g = c, x[i].copy(); best = decode(w, s, d, x[i])
        mb = pb.mean(0); alpha = 1.0 - 0.5 * it / P['ITER']
        for i in range(M):
            phi = rng.random(N); pt = phi * pb[i] + (1 - phi) * g
            u = rng.random(N) + 1e-9
            sgn = np.where(rng.random(N) < 0.5, 1, -1)
            x[i] = np.clip(pt + sgn * alpha * np.abs(mb - x[i]) * np.log(1 / u), 0, 1)
    return best

def disc_qaco(w, s, d, rng, st):
    N = P['N']; th = np.full((N, N), math.pi / 4)
    eta = np.where(w.adj, 1.0 / np.maximum(w.C, 1e-6), 0.0)
    best = dijkstra(w.C, s, d)
    if best is None: return None
    bc = path_cost(w, best)
    for _ in range(P['ITER']):
        ib, ibc = None, INF
        for _ in range(P['POP']):
            cur, vis, path = s, {s}, [s]
            while cur != d and len(path) < N:
                pr = (np.sin(th[cur]) ** 2) * (eta[cur] ** 2.0)
                for v in vis: pr[v] = 0
                tot = pr.sum()
                if tot <= 0: path = None; break
                cur = int(rng.choice(N, p=pr / tot)); vis.add(cur); path.append(cur)
            if path is None or path[-1] != d: continue
            c = path_cost(w, path)
            if c < ibc: ib, ibc = path, c
        th -= 0.02 * (th - math.pi / 4)  # relax
        if ib is not None:
            if ibc < bc: best, bc = ib, ibc
            for a, b in zip(ib[:-1], ib[1:]):
                th[a, b] = min(th[a, b] + 0.08 * math.pi, math.pi / 2 - 0.05)
    return best

DISC = dict(AODV=disc_aodv, DSR=disc_dsr, DijkstraC=disc_dijkstraC,
            QGA=disc_qga, QPSO=disc_qpso, QACO=disc_qaco)

# ------------------------------------------------------------------ MOR-Q
class MorqFlow:
    """Per-flow quantum-inspired probability state over K candidate paths."""
    def __init__(self, mob=True):
        self.mob = mob; self.cands = []; self.theta = np.array([])
    def refresh(self, w, s, d):
        C = w.C if self.mob else w.C_nomob
        self.cands = k_diverse(w, s, d, P['K_CAND'], C)
        costs = np.array([path_cost(w, p) for p in self.cands])
        if len(costs):
            prob = np.exp(-(costs - costs.min()) / 0.15); prob /= prob.sum()
            self.theta = np.arcsin(np.sqrt(prob))      # amplitude angles
        else:
            self.theta = np.array([])
    def probs(self, w, ok):
        pr = np.sin(self.theta) ** 2 * ok
        s = pr.sum()
        return pr / s if s > 0 else None
    def reward(self, k, r):
        self.theta[k] = np.clip(self.theta[k] + r * 0.10 * math.pi, 0.05, math.pi / 2 - 0.05)

# ------------------------------------------------------------------ simulation
def simulate(proto, seed, vmax):
    w = World(seed, vmax)
    rng = np.random.default_rng(seed * 7919 + 13)   # protocol-side randomness
    flows = w.flows
    route = {f: None for f in flows}; st = {f: {} for f in flows}
    mq = {f: MorqFlow(mob=(proto == 'MOR-Q')) for f in flows}
    sent = deliv = 0; energy = ctrl = 0.0; delays = []; breaks = disc_n = 0
    for t in range(P['T']):
        w.step()
        for f in flows:
            s, d = f; sent += 1; extra_delay = 0.0
            # ----- route selection
            if proto.startswith('MOR-Q'):
                q = mq[f]; mob = (proto == 'MOR-Q')
                def ok_mask():
                    m = np.array([1.0 if valid(w, p) and (not mob or min_lt(w, p) >= P['TAU_MIN']) else 0.0
                                  for p in q.cands]) if q.cands else np.array([])
                    return m
                ok = ok_mask()
                pr = q.probs(w, ok) if len(ok) else None
                if pr is None:                      # no usable candidate -> new discovery
                    ctrl += P['E_FLOOD'] * P['N']; disc_n += 1; extra_delay = P['DISC_MS']
                    q.refresh(w, s, d); ok = ok_mask(); pr = q.probs(w, ok)
                    if pr is None and len(ok): pr = np.ones(len(ok)) / len(ok)
                if pr is None: continue
                k = int(rng.choice(len(pr), p=pr)); path = q.cands[k]
            else:
                if route[f] is None:
                    ctrl += P['E_FLOOD'] * P['N']; disc_n += 1; extra_delay = P['DISC_MS']
                    route[f] = DISC[proto](w, s, d, rng, st[f])
                if route[f] is None: continue
                path = route[f]
            # ----- transmit along path
            ok_all = True; dly = extra_delay; e = 0.0
            for a, b in zip(path[:-1], path[1:]):
                if not w.adj[a, b]:               # link broken (node moved out of range)
                    ok_all = False; breaks += 1; e += P['E_ELEC'] / 2; break
                p = w.p[a, b]
                if rng.random() < p: att = 1; got = True
                elif rng.random() < p: att = 2; got = True
                else: att = 2; got = False
                e += att * w.E[a, b]
                dly += P['HOP_MS'] + (att - 1) * P['RETRY_MS'] + rng.exponential(2.0)
                if not got: ok_all = False; break
            energy += e
            if ok_all: deliv += 1; delays.append(dly)
            # ----- feedback
            if proto.startswith('MOR-Q'):
                q.reward(k, 1.0 if ok_all else -1.5)
            else:
                # only a link break (not a lossy hop) triggers a route error
                if not ok_all and any(not w.adj[a, b] for a, b in zip(path[:-1], path[1:])):
                    if proto == 'DSR' and st[f].get('cache'):
                        route[f] = st[f]['cache'].pop(0)   # cached route may itself be stale
                    else:
                        route[f] = None
    # periodic beacon overhead: HELLO for all, position/velocity beacons for mobility-aware cost
    mob_aware = proto in ('DijkstraC', 'QGA', 'QPSO', 'QACO', 'MOR-Q')
    ctrl += P['N'] * P['T'] * (P['BEACON_MOB'] if mob_aware else P['BEACON_HELLO'])
    tot_e = energy + ctrl
    return dict(proto=proto, seed=seed, vmax=vmax, sent=sent, delivered=deliv,
                pdr=deliv / max(sent, 1),
                energy_per_pkt_mJ=tot_e / max(deliv, 1),
                ctrl_share=ctrl / max(tot_e, 1e-9),
                delay_ms=float(np.mean(delays)) if delays else np.nan,
                route_breaks=breaks, discoveries=disc_n)

PROTOS = ['AODV', 'DSR', 'DijkstraC', 'QGA', 'QPSO', 'QACO', 'MOR-Q', 'MOR-Q-noMob']

if __name__ == '__main__':
    speeds = [2, 5, 10, 15, 20]
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
    os.makedirs('results', exist_ok=True)
    rows = []; t0 = time.time()
    for v in speeds:
        for sd in seeds:
            for pr in PROTOS:
                rows.append(simulate(pr, sd, v))
        print(f'speed {v} done, {time.time()-t0:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv('results/raw_results.csv', index=False)
