"""
NetCore: AI-Based Predictive Network Congestion Management in Data Centers
==========================================================================
Core engine. Contents (search for these names):

  dijkstra()           - hand-written Dijkstra (binary heap)            [Algorithm]
  build_topology()     - leaf-spine data-center graph
  make_scenario()      - 5 traffic scenarios
  Sim                  - 1-second network simulator + controllers
      .step()          - network physics (utilisation, queue, latency, loss)
      .control()       - reactive trigger  vs  ML-predicted trigger
      .best_path()     - congestion-aware Dijkstra, Eq. (1)
  features()           - short-term-history feature engineering         [ML]
  train_all()          - Random Forest / Logistic Reg. / Decision Tree / XGBoost
  evaluate(), stats    - multi-seed evaluation, 95 % CIs, paired t-tests
  main()               - python netcore.py  -> results/ folder

Controllers compared:  ecmp (static hash, no rerouting)  |  spray (ideal per-packet
ECMP upper-bound reference)  |  reactive (threshold)  |  netcore (ML-predictive)
"""
import os, time, json, math, heapq, argparse, warnings
import numpy as np
import networkx as nx
import pandas as pd
import joblib
from joblib import Parallel, delayed
from scipy import stats as sps
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "cache", "models.joblib")

# ----------------------------------------------------------------- constants
T = 300            # seconds per episode (1 sample / second)
H = 8              # prediction horizon (s): "congestion within next 5-10 s"
CONG = 0.90        # a link is "congested" when utilisation > 90 %
P_THR = 0.5        # probability threshold that triggers proactive rerouting
COOLDOWN = 5       # s before the same flow may be rerouted again
WEIGHTS = (0.5, 0.2, 0.2, 0.1)      # Eq. (1): alpha, beta, gamma, delta
SCENARIOS = ["steady", "burst", "hotspot", "repeated", "multi"]
CONTROLLERS = ["ecmp", "spray", "reactive", "netcore"]
LABEL = {"ecmp": "ECMP (static hash)", "spray": "Ideal spray (reference)",
         "reactive": "Reactive threshold", "netcore": "NetCore (ML-predictive)"}
COLOR = {"ecmp": "#9e9e9e", "spray": "#7cb342", "reactive": "#e07b39", "netcore": "#2a7ab0"}
SPINES = ["SW3", "SW4", "SW5"]
FEATS = ["util", "traffic_mbps", "pkt_rate", "latency", "loss", "throughput",
         "queue", "active_flows", "util_mean5", "lat_mean5", "queue_mean5",
         "util_trend", "lat_trend", "queue_trend", "util_max5"]
RAW_COLS = list(range(8))          # the 8 instantaneous parameters (no history)


# ======================================================================
#  1.  DIJKSTRA  (hand-written, binary-heap implementation)
# ======================================================================
def dijkstra(adj, src, dst, weight):
    """Dijkstra's shortest-path algorithm.

    adj    : dict node -> iterable of neighbour nodes
    weight : function (u, v) -> non-negative edge cost  (here: Eq. (1))
    Returns (path, cost); (None, inf) if dst is unreachable.
    """
    dist = {src: 0.0}
    prev = {}
    heap = [(0.0, src)]                 # priority queue of (distance, node)
    visited = set()
    while heap:
        d, u = heapq.heappop(heap)      # node with the smallest tentative distance
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            break
        for v in adj[u]:                # relax every outgoing edge
            if v in visited:
                continue
            nd = d + weight(u, v)
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if dst not in visited:
        return None, math.inf
    path = [dst]
    while path[-1] != src:              # walk predecessors back to the source
        path.append(prev[path[-1]])
    return path[::-1], dist[dst]


# ======================================================================
#  2.  TOPOLOGY + TRAFFIC
# ======================================================================
def build_topology():
    """Leaf-spine: 8 servers, 2 leaf switches, 3 spine switches (5 switches)."""
    G = nx.DiGraph()
    left, right = ["S1", "S2", "S3", "S4"], ["S5", "S6", "S7", "S8"]
    for leaf, srv in (("SW1", left), ("SW2", right)):
        for s in srv:
            for a, b in ((s, leaf), (leaf, s)):
                G.add_edge(a, b, cap=2000.0, base=0.05)
    for leaf in ("SW1", "SW2"):
        for sp in SPINES:
            for a, b in ((leaf, sp), (sp, leaf)):
                G.add_edge(a, b, cap=1000.0, base=0.10)
    return G, left, right


def bump(t, start, rise, hold, fall):
    return np.interp(t, [start, start + rise, start + rise + hold,
                         start + rise + hold + fall], [0, 1, 1, 0], left=0, right=0)


def make_scenario(name, seed):
    """Returns flows, rate matrix [T,F] (Mbps), initial spine index per flow."""
    rng = np.random.default_rng(seed)
    left, right = ["S1", "S2", "S3", "S4"], ["S5", "S6", "S7", "S8"]
    flows = [(s, str(rng.choice(right))) for s in left for _ in range(2)] + \
            [(s, str(rng.choice(left))) for s in right for _ in range(2)]
    F = len(flows)
    t = np.arange(T)
    peak = 337.0 * rng.uniform(0.9, 1.1, F)          # ~90 % of a balanced fabric
    init = rng.integers(0, 3, F)                      # ECMP-style hash placement
    mult = np.ones((T, F))
    amp = rng.uniform(0.7, 1.0)
    if name == "steady":                              # slow ramp 45 % -> 95 %
        level = np.linspace(0.45, 0.95, T) * rng.uniform(0.92, 1.05)
    elif name == "burst":                             # one sudden burst on 6 flows
        level = np.full(T, 0.55 * rng.uniform(0.9, 1.05))
        sel = rng.choice(F, 6, replace=False)
        st = int(rng.integers(90, 160))
        mult[:, sel] = 1 + amp * bump(t, st, 4, 40, 8)[:, None]
    elif name == "hotspot":                           # 5 flows pinned to one spine, growing
        level = np.full(T, 0.5 * rng.uniform(0.92, 1.05))
        sel = list(range(5)); init[sel] = 0
        st = int(rng.integers(40, 90))
        mult[:, sel] = 1 + 0.9 * amp * bump(t, st, 60, 60, 40)[:, None]
    elif name == "repeated":                          # 4 bursts of 12 s on random flows
        level = np.full(T, 0.55 * rng.uniform(0.9, 1.05))
        for k in range(4):
            sel = rng.choice(F, 5, replace=False)
            mult[:, sel] *= 1 + amp * bump(t, 40 + 65 * k + int(rng.integers(0, 10)), 5, 12, 5)[:, None]
    elif name == "multi":                             # two links overloaded at once
        level = np.full(T, 0.5 * rng.uniform(0.92, 1.05))
        a, b = [0, 1, 2, 3], [8, 9, 10, 11]
        init[a] = 0; init[b] = 1
        mult[:, a] = 1 + 0.9 * amp * bump(t, 70, 10, 50, 20)[:, None]
        mult[:, b] = 1 + 0.9 * amp * bump(t, 90, 10, 50, 20)[:, None]
    else:
        raise ValueError(name)
    noise = np.zeros((T, F)); e = rng.normal(0, 0.03, (T, F))
    for i in range(1, T):
        noise[i] = 0.8 * noise[i - 1] + e[i]
    rates = np.clip(peak * level[:, None] * mult * (1 + noise), 0, None)
    return flows, rates, init


# ======================================================================
#  3.  SIMULATOR + CONTROLLERS
# ======================================================================
class Sim:
    def __init__(self, scn, seed, model=None, controller="ecmp"):
        self.G, _, _ = build_topology()
        self.adj = {u: list(self.G.successors(u)) for u in self.G.nodes}
        self.links = list(self.G.edges())
        self.idx = {e: i for i, e in enumerate(self.links)}
        self.L = len(self.links)
        self.cap = np.array([self.G.edges[e]["cap"] for e in self.links])
        self.base = np.array([self.G.edges[e]["base"] for e in self.links])
        self.flows, self.rates, init = make_scenario(scn, seed)
        self.rng = np.random.default_rng(seed + 12345)   # measurement noise
        self.paths = []
        for (s, d), sp in zip(self.flows, init):
            ls = "SW1" if s in ("S1", "S2", "S3", "S4") else "SW2"
            ld = "SW2" if ls == "SW1" else "SW1"
            self.paths.append([s, ls, SPINES[sp], ld, d])
        if controller == "spray":        # ideal per-packet ECMP: every flow split over all spines
            nf, npth = [], []
            for fl, p in zip(self.flows, self.paths):
                for sp in SPINES:
                    nf.append(fl); npth.append([p[0], p[1], sp, p[3], p[4]])
            self.flows, self.paths = nf, npth
            self.rates = np.repeat(self.rates / 3, 3, axis=1)
        self.F = len(self.flows)
        self.scn, self.seed = scn, seed
        self.controller, self.model = controller, model
        # tunables (ablations change these)
        self.lag, self.p_thr, self.weights, self.use_nx = 0, P_THR, WEIGHTS, False
        # state / logs
        self.q = np.zeros(self.L)
        self.M = np.zeros((T, self.L, 8))       # observed metrics (noisy)
        self.true_util = np.zeros((T, self.L))
        self.prob = np.zeros((T, self.L))       # RF congestion probability per link
        self.place = np.zeros((T, self.F), int)  # spine used by every flow over time
        self.events = []                         # reroute log
        self.last_move = np.full(self.F, -99)
        self.reroutes, self.dec_times = 0, []
        self.fl_lat, self.offered, self.deliv = [], 0.0, 0.0
        self.cong_ls, self.onsets, self.maxutil = 0, 0, np.zeros(T)
        self.cong_prev = np.zeros(self.L, bool)
        self.avg_lat_t = np.zeros(T); self.loss_t = np.zeros(T); self.thr_t = np.zeros(T)

    def plinks(self, path):
        return [self.idx[(a, b)] for a, b in zip(path[:-1], path[1:])]

    # -- one second of network dynamics --------------------------------------
    def step(self, t):
        load = np.zeros(self.L); nfl = np.zeros(self.L)
        pl = [self.plinks(p) for p in self.paths]
        for f, l in enumerate(pl):
            load[l] += self.rates[t, f]; nfl[l] += 1
        util = load / self.cap
        self.q = np.clip(0.7 * self.q + np.maximum(0, util - 0.7) * 400, 0, 500)
        lat = self.base + 0.02 * self.q + 0.3 * util ** 4
        over = np.maximum(0, util - 1) / np.maximum(util, 1e-9)
        n = lambda s: self.rng.normal(0, s, self.L)
        self.M[t] = np.stack([
            util * (1 + n(.02)), load * (1 + n(.02)), load * 125 * (1 + n(.02)),
            np.clip(lat + n(.05), 0, None), over * 100 + np.abs(n(.02)),
            load * (1 - over) * (1 + n(.02)), self.q * (1 + n(.03)), nfl], axis=1)
        self.true_util[t] = util
        self.place[t] = [SPINES.index(p[2]) for p in self.paths]
        fl_l = np.array([lat[l].sum() for l in pl])
        fl_loss = np.array([1 - np.prod(1 - over[l]) for l in pl])
        self.fl_lat.extend(fl_l.tolist())
        off, dlv = self.rates[t].sum(), (self.rates[t] * (1 - fl_loss)).sum()
        self.offered += off; self.deliv += dlv
        self.avg_lat_t[t] = (fl_l * self.rates[t]).sum() / max(off, 1e-9)
        self.loss_t[t] = 100 * (1 - dlv / max(off, 1e-9)); self.thr_t[t] = dlv
        cong = util > CONG
        self.cong_ls += int(cong.sum()); self.onsets += int((cong & ~self.cong_prev).sum())
        self.cong_prev = cong; self.maxutil[t] = util.max()
        self.pl, self.load = pl, load

    # -- control ---------------------------------------------------------------
    def control(self, t):
        t0 = time.perf_counter()
        tt = max(0, t - self.lag)          # controller only sees (possibly stale) telemetry
        obs = self.M[tt]
        if self.controller == "reactive":
            trig = obs[:, 0] > CONG                       # congestion already happening
        else:                                              # netcore: predicted congestion
            p = self.model.predict_proba(features(self.M, tt))[:, 1]
            self.prob[t] = p
            trig = p >= self.p_thr
        # controller knows its own routing table: expected load = flow rates on CURRENT paths
        cur = np.zeros(self.L)
        for f in range(self.F):
            for k in self.pl[f]: cur[k] += self.rates[tt, f]
        moved = False
        for l in sorted(np.where(trig)[0], key=lambda i: -obs[i, 0]):
            cands = [f for f in range(self.F) if l in self.pl[f]]
            for f in sorted(cands, key=lambda f: -self.rates[tt, f]):
                if t - self.last_move[f] < COOLDOWN + self.lag:
                    continue
                new = self.best_path(f, tt, cur, obs)
                if new is not None:
                    old = self.paths[f]
                    for k in self.pl[f]: cur[k] -= self.rates[tt, f]
                    self.paths[f] = new
                    for k in self.plinks(new): cur[k] += self.rates[tt, f]
                    self.pl[f] = self.plinks(new)
                    self.last_move[f] = t; self.reroutes += 1; moved = True
                    self.events.append(dict(
                        t=t, flow=f"{self.flows[f][0]}->{self.flows[f][1]}",
                        trigger_link=f"{self.links[l][0]}->{self.links[l][1]}",
                        link_util_pct=round(100 * float(obs[l, 0]), 1),
                        old_path=" > ".join(old), new_path=" > ".join(new)))
                    break                                  # one flow per link per second
        if moved or trig.any():
            self.dec_times.append((time.perf_counter() - t0) * 1000)

    def parts(self, i, util_after, obs):
        """The four weighted terms of Eq. (1) for link i."""
        a, b, g, d = self.weights
        return (a * min(util_after, 1.5), b * min(obs[i, 3] / 20, 1.5),
                g * min(obs[i, 4] / 5, 1.0), d * 0.25)

    def best_path(self, f, tt, cur, obs):
        """Congestion-aware Dijkstra:  Cost = a*Util + b*Latency + g*Loss + d*Hops."""
        r = self.rates[tt, f]; mine = set(self.pl[f]); s, d = self.flows[f]

        def w(u, v):
            i = self.idx[(u, v)]
            util_after = (cur[i] - (r if i in mine else 0) + r) / self.cap[i]
            return sum(self.parts(i, util_after, obs))
        if self.use_nx:                                    # optional cross-check
            new = nx.dijkstra_path(self.G, s, d, weight=lambda u, v, _: w(u, v))
        else:
            new, _ = dijkstra(self.adj, s, d, w)          # <- OUR Dijkstra
        if new is None or new == self.paths[f]:
            return None
        new_bott = max((cur[i] + (0 if i in mine else r)) / self.cap[i] for i in self.plinks(new))
        old_bott = max(cur[i] / self.cap[i] for i in self.pl[f])
        return new if new_bott < old_bott - 0.05 else None   # move only if clearly better

    def run(self):
        for t in range(T):
            self.step(t)
            if self.controller in ("reactive", "netcore"):
                self.control(t)
        return self.summary()

    def summary(self):
        lat = np.array(self.fl_lat)
        return dict(avg_latency=float(lat.mean()), p95_latency=float(np.percentile(lat, 95)),
                    packet_loss=float(100 * (1 - self.deliv / self.offered)),
                    throughput=float(self.deliv / T), cong_link_seconds=int(self.cong_ls),
                    cong_events=int(self.onsets), reroutes=int(self.reroutes),
                    decision_ms=float(np.mean(self.dec_times)) if self.dec_times else 0.0)


def candidate_paths(sim, src, dst, t, rate, weights=None):
    """Explain Eq. (1) for every loop-free src->dst path at time t (used by the dashboard)."""
    saved = sim.weights
    if weights is not None:
        sim.weights = weights
    try:
        return _candidate_paths(sim, src, dst, t, rate)
    finally:
        sim.weights = saved                  # never leave the simulator modified


def _candidate_paths(sim, src, dst, t, rate):
    obs, cur = sim.M[t], sim.M[t, :, 1]

    def w(u, v):
        i = sim.idx[(u, v)]
        return sum(sim.parts(i, (cur[i] + rate) / sim.cap[i], obs))
    best, _ = dijkstra(sim.adj, src, dst, w)
    nxp = nx.dijkstra_path(sim.G, src, dst, weight=lambda u, v, _: w(u, v))
    rows = []
    for p in nx.all_simple_paths(sim.G, src, dst):
        comp = np.zeros(4)
        for u, v in zip(p[:-1], p[1:]):
            i = sim.idx[(u, v)]
            comp += sim.parts(i, (cur[i] + rate) / sim.cap[i], obs)
        rows.append(dict(path=" > ".join(p), util_term=comp[0], latency_term=comp[1],
                         loss_term=comp[2], hop_term=comp[3], total=comp.sum(), chosen=(p == best)))
    return pd.DataFrame(rows).sort_values("total").reset_index(drop=True), best, nxp


# ======================================================================
#  4.  MACHINE LEARNING
# ======================================================================
def features(M, t):
    """15 features per link: 8 raw + rolling means/trend/max over the last few seconds."""
    cur = M[t]; w = M[max(0, t - 4):t + 1]; prev = M[max(0, t - 3)]
    cols = [0, 3, 6]
    return np.hstack([cur, w.mean(0)[:, cols], cur[:, cols] - prev[:, cols],
                      w[:, :, 0].max(0)[:, None]])


def build_dataset(seeds, horizon=H):
    X, y, grp, now = [], [], [], []
    for k, (scn, seed) in enumerate(seeds):
        s = Sim(scn, seed); s.run()
        cong = s.true_util > CONG
        for t in range(T - horizon):
            X.append(features(s.M, t))
            y.append(cong[t + 1:t + horizon + 1].any(0))
            now.append(cong[t]); grp += [k] * s.L
    return np.vstack(X), np.concatenate(y).astype(int), np.array(grp), np.concatenate(now)


class ColSel:
    """Wrap a model so it only sees a subset of feature columns (feature ablation)."""
    def __init__(self, model, cols): self.model, self.cols = model, cols
    def fit(self, X, y): self.model.fit(X[:, self.cols], y); return self
    def predict_proba(self, X): return self.model.predict_proba(X[:, self.cols])


def gbm_model():
    """XGBoost if it can be loaded, else scikit-learn's gradient boosting."""
    try:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.1, subsample=0.8,
                          n_jobs=-1, eval_metric="logloss", verbosity=0, random_state=0)
        return "XGBoost", m
    except Exception:
        return "GradBoost (XGBoost unavailable)", HistGradientBoostingClassifier(
            max_depth=6, max_iter=120, random_state=0)


def new_models():
    gname, gm = gbm_model()
    return {"Random Forest": RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=20,
                                                    n_jobs=-1, random_state=0),
            "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
            "Decision Tree": DecisionTreeClassifier(max_depth=14, min_samples_leaf=20, random_state=0),
            gname: gm}


def clf_metrics(yte, p, now_te, thr=P_THR):
    pred = (p >= thr).astype(int); early = ~now_te.astype(bool)
    return dict(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred, zero_division=0),
                recall=recall_score(yte, pred), f1=f1_score(yte, pred), auc=roc_auc_score(yte, p),
                early_recall=recall_score(yte[early], pred[early]),
                early_precision=precision_score(yte[early], pred[early], zero_division=0),
                early_f1=f1_score(yte[early], pred[early]),
                cm=confusion_matrix(yte, pred).tolist())


def train_ids(): return [(sc, 1000 + i * 10 + j) for j, sc in enumerate(SCENARIOS) for i in range(8)]
def test_ids():  return [(sc, 2000 + i * 10 + j) for j, sc in enumerate(SCENARIOS) for i in range(3)]


def train_all(force=False, verbose=True):
    """Train (or load from cache) all predictors. Train/test episodes use different seeds."""
    if os.path.exists(CACHE) and not force:
        return joblib.load(CACHE)
    t0 = time.time()
    Xtr, ytr, _, _ = build_dataset(train_ids()); Xte, yte, _, now_te = build_dataset(test_ids())
    if verbose:
        print(f"train rows {len(ytr)} (pos {ytr.mean():.2%}) | test rows {len(yte)} (pos {yte.mean():.2%})")
    models, clf, probs = new_models(), {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr); probs[name] = m.predict_proba(Xte)[:, 1]
        clf[name] = clf_metrics(yte, probs[name], now_te)
        if verbose:
            print(f"  {name:32s}", {k: round(v, 3) for k, v in clf[name].items() if k != "cm"})
    rf = models["Random Forest"]
    imp = dict(zip(FEATS, rf.feature_importances_.tolist()))
    for m in models.values():                      # single-thread inference = realistic latency
        if hasattr(m, "n_jobs"): m.n_jobs = 1
    out = dict(models=models, clf=clf, importances=imp,
               test=dict(y=yte, now=now_te, p=probs), seconds=time.time() - t0)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    joblib.dump(out, CACHE)
    return out


# ======================================================================
#  5.  EVALUATION + STATISTICS
# ======================================================================
def run_one(scn, seed, controller, model=None, **kw):
    s = Sim(scn, seed, model if controller == "netcore" else None, controller)
    for k, v in kw.items(): setattr(s, k, v)
    return s.run()


def evaluate(model, scenarios, seeds, controllers, n_jobs=1, **kw):
    """res[scenario][controller] = list of summaries (one per seed, same traffic for all)."""
    jobs = [(sc, sd, c) for sc in scenarios for c in controllers for sd in seeds]
    out = Parallel(n_jobs=n_jobs)(delayed(run_one)(sc, sd, c, model, **kw) for sc, sd, c in jobs)
    res = {sc: {c: [] for c in controllers} for sc in scenarios}
    for (sc, sd, c), r in zip(jobs, out): res[sc][c].append(r)
    return res


def mean_ci(x, level=0.95):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 2: return m, 0.0
    return m, float(sps.t.ppf((1 + level) / 2, n - 1) * x.std(ddof=1) / math.sqrt(n))


def paired_test(a, b):
    """Paired t-test of a-b over seeds. Returns mean diff, 95 % CI, p-value."""
    d = np.asarray(a, float) - np.asarray(b, float)
    m, h = mean_ci(d)
    p = 1.0 if np.allclose(d, 0) else float(sps.ttest_rel(a, b).pvalue)
    return m, h, p


METRICS = ["avg_latency", "p95_latency", "packet_loss", "throughput",
           "cong_link_seconds", "cong_events", "reroutes", "decision_ms"]


def results_tables(res):
    rows = []
    for sc in res:
        for c in res[sc]:
            for k in METRICS:
                m, h = mean_ci([r[k] for r in res[sc][c]])
                rows.append(dict(scenario=sc, controller=c, metric=k, mean=m, ci95=h, n=len(res[sc][c])))
    ci = pd.DataFrame(rows)
    prow = []
    for k in ("avg_latency", "p95_latency", "packet_loss", "cong_link_seconds", "cong_events"):
        allr = {c: [] for c in ("netcore", "reactive")}
        for sc in res:
            a = [r[k] for r in res[sc]["netcore"]]; b = [r[k] for r in res[sc]["reactive"]]
            m, h, p = paired_test(a, b)
            prow.append(dict(scenario=sc, metric=k, diff_netcore_minus_reactive=m,
                             ci_lo=m - h, ci_hi=m + h, p_value=p))
            allr["netcore"] += a; allr["reactive"] += b
        m, h, p = paired_test(allr["netcore"], allr["reactive"])
        prow.append(dict(scenario="ALL", metric=k, diff_netcore_minus_reactive=m,
                         ci_lo=m - h, ci_hi=m + h, p_value=p))
    return ci, pd.DataFrame(prow)


# ======================================================================
#  6.  FIGURES
# ======================================================================
def make_figs(models_out, res, outdir=RES):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clf, imp = models_out["clf"], models_out["importances"]
    cm = np.array(clf["Random Forest"]["cm"])
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    sim = Sim("burst", 3001); sim.run()
    ax[0, 0].scatter(sim.true_util.ravel()[::3] * 100, sim.M[:, :, 3].ravel()[::3], s=4, alpha=.4)
    ax[0, 0].axvline(90, color="r", ls="--")
    ax[0, 0].set(xlabel="Link utilisation (%)", ylabel="Latency (ms)", title="(a) Utilisation vs latency")
    x = np.arange(len(SCENARIOS)); wd = 0.2

    def bars(a, key, title, yl):
        for i, c in enumerate(CONTROLLERS):
            ms = [mean_ci([r[key] for r in res[s][c]]) for s in SCENARIOS]
            a.bar(x + (i - 1.5) * wd, [m for m, _ in ms], wd, yerr=[h for _, h in ms],
                  label=LABEL[c], color=COLOR[c], capsize=2)
        a.set_xticks(x); a.set_xticklabels(SCENARIOS); a.set(title=title, ylabel=yl)
    bars(ax[0, 1], "throughput", "(b) Throughput (mean, 95% CI)", "Mbps")
    bars(ax[0, 2], "cong_link_seconds", "(c) Congested link-seconds", "link-seconds")
    ax[0, 2].legend(fontsize=7)
    ax[1, 0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): ax[1, 0].text(j, i, f"{cm[i, j]}", ha="center", va="center")
    ax[1, 0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No cong.", "Cong."],
                 yticklabels=["No cong.", "Cong."], xlabel="Predicted", ylabel="Actual",
                 title="(d) RF confusion matrix")
    bars(ax[1, 1], "p95_latency", "(e) 95th-percentile latency", "ms")
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:8][::-1]
    ax[1, 2].barh([k for k, _ in top], [v for _, v in top], color="#2a7ab0")
    ax[1, 2].set(title="(f) RF feature importance")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_results.png"), dpi=130); plt.close()

    rf = models_out["models"]["Random Forest"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    for a, sc, sd in zip(ax, ("burst", "hotspot"), (3001, 3002)):
        for c in ("ecmp", "reactive", "netcore"):
            s = Sim(sc, sd, rf if c == "netcore" else None, c); s.run()
            a.plot(s.maxutil * 100, label=LABEL[c], color=COLOR[c])
        a.axhline(90, color="r", ls="--", lw=.8)
        a.set(title=f"Worst-link utilisation - {sc}", xlabel="time (s)", ylabel="%"); a.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_timeseries.png"), dpi=130); plt.close()


def write_md(models_out, ci, paired, path):
    clf = models_out["clf"]
    L = ["## Predictor (held-out episodes, threshold 0.5)\n",
         "| Model | Acc | Prec | Recall | F1 | AUC | Early-warning recall | Early-warning precision |",
         "|---|---|---|---|---|---|---|---|"]
    for n, c in clf.items():
        L.append(f"| {n} | {c['accuracy']:.3f} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | "
                 f"{c['auc']:.3f} | {c['early_recall']:.3f} | {c['early_precision']:.3f} |")
    L += ["\n## End-to-end (mean +/- 95% CI over seeds)\n",
          "| Scenario | Method | Avg lat (ms) | P95 lat (ms) | Loss (%) | Thr (Mbps) | Cong. events | Cong. link-s | Reroutes |",
          "|---|---|---|---|---|---|---|---|---|"]
    g = ci.set_index(["scenario", "controller", "metric"])
    f = lambda s, c, k, d=2: f"{g.loc[(s, c, k), 'mean']:.{d}f} +/- {g.loc[(s, c, k), 'ci95']:.{d}f}"
    for sc in SCENARIOS:
        for c in CONTROLLERS:
            L.append(f"| {sc} | {c} | {f(sc,c,'avg_latency')} | {f(sc,c,'p95_latency')} | {f(sc,c,'packet_loss',3)} | "
                     f"{f(sc,c,'throughput',0)} | {f(sc,c,'cong_events',1)} | {f(sc,c,'cong_link_seconds',1)} | "
                     f"{f(sc,c,'reroutes',1)} |")
    L += ["\n## NetCore minus Reactive (paired over seeds; negative = NetCore better)\n",
          "| Scenario | Metric | Mean diff | 95% CI | p-value |", "|---|---|---|---|---|"]
    for _, r in paired.iterrows():
        L.append(f"| {r.scenario} | {r.metric} | {r.diff_netcore_minus_reactive:.3f} | "
                 f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}] | {r.p_value:.3f} |")
    L += ["\n## RF feature importance\n"] + [f"- {k}: {v:.3f}" for k, v in
                                             sorted(models_out["importances"].items(), key=lambda kv: -kv[1])]
    open(path, "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20, help="evaluation seeds per scenario")
    ap.add_argument("--retrain", action="store_true", help="ignore cached models")
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers (-1 = all cores)")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True); t0 = time.time()
    mo = train_all(force=a.retrain)
    rf = mo["models"]["Random Forest"]
    seeds = list(range(3000, 3000 + a.seeds))
    res = evaluate(rf, SCENARIOS, seeds, CONTROLLERS, n_jobs=a.jobs)
    ci, paired = results_tables(res)
    ci.to_csv(os.path.join(RES, "results_ci.csv"), index=False)
    paired.to_csv(os.path.join(RES, "paired_tests.csv"), index=False)
    json.dump(dict(classifier=mo["clf"], importances=mo["importances"]),
              open(os.path.join(RES, "predictor.json"), "w"), indent=1)
    write_md(mo, ci, paired, os.path.join(RES, "results.md"))
    make_figs(mo, res)
    print(f"done in {time.time() - t0:.0f}s -> {RES}")


if __name__ == "__main__":
    main()
