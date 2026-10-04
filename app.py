"""
NetCore dashboard.   Run:  streamlit run app.py
Tabs: live animated topology | method comparison | predictor | Dijkstra explorer | experiments | about
"""
import os, inspect
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, precision_recall_curve
import netcore as nc

st.set_page_config(page_title="NetCore", page_icon="🌐", layout="wide")


# ---------------------------------------------------------------- small helpers
def pc(fig, **kw):
    try: st.plotly_chart(fig, width="stretch", **kw)
    except TypeError: st.plotly_chart(fig, use_container_width=True, **kw)


def dfs(df, **kw):
    try: st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError: st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def img(path):
    try: st.image(path, width="stretch")
    except TypeError: st.image(path, use_container_width=True)


@st.cache_resource(show_spinner="Training predictors (one-time, about 1-2 minutes)...")
def load_models():
    return nc.train_all(verbose=False)


@st.cache_resource(show_spinner=False, max_entries=60)
def simulate(scn, seed, controller, model_name, p_thr, weights, lag):
    mo = load_models()
    sim = nc.Sim(scn, seed, mo["models"][model_name] if controller == "netcore" else None, controller)
    sim.p_thr, sim.weights, sim.lag = p_thr, weights, lag
    sim.run()
    return sim


@st.cache_data(show_spinner=False)
def unmanaged_probs(scn, seed, model_name):
    """Run the predictor on traffic with NO rerouting to see how well it warns ahead."""
    mo = load_models(); sim = simulate(scn, seed, "ecmp", model_name, nc.P_THR, nc.WEIGHTS, 0)
    m = mo["models"][model_name]
    return np.array([m.predict_proba(nc.features(sim.M, t))[:, 1] for t in range(nc.T)])


@st.cache_data(show_spinner=False)
def multi_seed(scn, n, model_name, p_thr, weights, lag):
    mo = load_models()
    res = nc.evaluate(mo["models"][model_name], [scn], list(range(3000, 3000 + n)), nc.CONTROLLERS,
                      lag=lag, p_thr=p_thr, weights=weights)
    rows = []
    for c in nc.CONTROLLERS:
        row = {"method": nc.LABEL[c]}
        for k in ("avg_latency", "p95_latency", "packet_loss", "throughput", "cong_link_seconds", "reroutes"):
            m, h = nc.mean_ci([r[k] for r in res[scn][c]]); row[k] = f"{m:.2f} ± {h:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- topology animation
POS = {"S1": (0, 0), "S2": (1, 0), "S3": (2, 0), "S4": (3, 0), "S5": (5, 0), "S6": (6, 0), "S7": (7, 0),
       "S8": (8, 0), "SW1": (1.5, 1.1), "SW2": (6.5, 1.1), "SW3": (2.6, 2.3), "SW4": (4.25, 2.3), "SW5": (5.9, 2.3)}
_LUT = px.colors.sample_colorscale("RdYlGn_r", [min(1, max(0, (i / 100 - 0.3) / 0.75)) for i in range(121)])
col = lambda u: _LUT[int(min(max(u, 0), 1.2) * 100)]


def link_geometry(links):
    g = []
    for u, v in links:
        (x0, y0), (x1, y1) = POS[u], POS[v]
        dx, dy = x1 - x0, y1 - y0; n = np.hypot(dx, dy)
        ox, oy = -dy / n * 0.07, dx / n * 0.07
        g.append((x0 + ox, y0 + oy, x1 + ox, y1 + oy))
    return g


def topology_figure(sim, step=2):
    geo = link_geometry(sim.links); L = sim.L
    ev = sim.events

    def title(t):
        done = [e for e in ev if e["t"] <= t]
        last = (f" | last reroute: {done[-1]['flow']} → {done[-1]['new_path'].split(' > ')[2]}"
                if done else "")
        return (f"<b>t = {t}s</b> | worst link {100 * sim.maxutil[t]:.0f}% | "
                f"reroutes so far: {len(done)}{last}")

    def frame_data(t):
        u = sim.true_util[t]
        lines = [go.Scatter(line=dict(color=col(u[i]), width=2 + 5 * min(u[i], 1.2))) for i in range(L)]
        hover = [f"{sim.links[i][0]}→{sim.links[i][1]}: {100 * u[i]:.0f}% "
                 f"({u[i] * sim.cap[i]:.0f}/{sim.cap[i]:.0f} Mbps)" for i in range(L)]
        mids = go.Scatter(marker=dict(color=[col(x) for x in u], size=9), text=hover)
        return lines + [mids]

    t0 = 0
    d0 = frame_data(t0)
    traces = []
    for i, (x0, y0, x1, y1) in enumerate(geo):
        traces.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip",
                                 line=d0[i].line, showlegend=False))
    traces.append(go.Scatter(x=[(g[0] + g[2]) / 2 for g in geo], y=[(g[1] + g[3]) / 2 for g in geo],
                             mode="markers", marker=d0[L].marker, text=d0[L].text,
                             hovertemplate="%{text}<extra></extra>", showlegend=False))
    kind = lambda n: "server" if n.startswith("S") and not n.startswith("SW") else ("leaf" if n in ("SW1", "SW2") else "spine")
    style = {"server": ("square", "#90caf9", 16), "leaf": ("diamond", "#ffcc80", 22), "spine": ("hexagon", "#ce93d8", 26)}
    for k, (sym, c, sz) in style.items():
        ns = [n for n in POS if kind(n) == k]
        traces.append(go.Scatter(x=[POS[n][0] for n in ns], y=[POS[n][1] for n in ns], mode="markers+text",
                                 text=ns, textposition="top center" if k != "server" else "bottom center",
                                 marker=dict(symbol=sym, size=sz, color=c, line=dict(color="#333", width=1)),
                                 hoverinfo="skip", showlegend=False))
    frames = [go.Frame(data=frame_data(t), traces=list(range(L + 1)), name=str(t),
                       layout=go.Layout(title_text=title(t))) for t in range(0, nc.T, step)]
    fig = go.Figure(data=traces, frames=frames)
    anim = lambda dur: dict(frame=dict(duration=dur, redraw=True), transition=dict(duration=0), mode="immediate")
    fig.update_layout(
        title_text=title(0), height=520, margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(visible=False, range=[-0.7, 8.7]), yaxis=dict(visible=False, range=[-0.6, 2.9]),
        updatemenus=[dict(type="buttons", x=0.02, y=-0.02, direction="left", buttons=[
            dict(label="▶ Play", method="animate", args=[None, anim(90)]),
            dict(label="⏸ Pause", method="animate", args=[[None], anim(0)])])],
        sliders=[dict(active=0, x=0.18, len=0.8, y=-0.02, currentvalue=dict(prefix="time (s): "),
                      steps=[dict(method="animate", label=str(t), args=[[str(t)], anim(0)])
                             for t in range(0, nc.T, step)])])
    return fig


# ---------------------------------------------------------------- sidebar
mo = load_models()
st.sidebar.title("🌐 NetCore")
st.sidebar.caption("AI-based predictive congestion management for data-center networks")
scn = st.sidebar.selectbox("Traffic scenario", nc.SCENARIOS, index=1,
                           help="steady ramp | sudden burst | single hotspot | repeated bursts | multi-link")
seed = int(st.sidebar.number_input("Random seed", 3000, 9999, 3001))
model_name = st.sidebar.selectbox("Predictor model", list(mo["models"].keys()))
p_thr = st.sidebar.slider("Trigger probability threshold", 0.1, 0.9, nc.P_THR, 0.05)
st.sidebar.markdown("**Eq. (1) weights**  `Cost = α·Util + β·Lat + γ·Loss + δ·Hops`")
wa = st.sidebar.slider("α utilisation", 0.0, 1.0, nc.WEIGHTS[0], 0.05)
wb = st.sidebar.slider("β latency", 0.0, 1.0, nc.WEIGHTS[1], 0.05)
wg = st.sidebar.slider("γ packet loss", 0.0, 1.0, nc.WEIGHTS[2], 0.05)
wd = st.sidebar.slider("δ hop count", 0.0, 1.0, nc.WEIGHTS[3], 0.05)
lag = st.sidebar.slider("Telemetry / control delay (s)", 0, 8, 0)
W = (wa, wb, wg, wd)
sim_args = (scn, seed, model_name, p_thr, W, lag)

st.title("NetCore — Predictive Congestion Management")
t1, t2, t3, t4, t5, t6 = st.tabs(["🕸️ Live topology", "⚖️ Compare methods", "🧠 Predictor",
                                  "🧮 Dijkstra explorer", "📊 Experiments", "ℹ️ About"])

# ================================================================ TAB 1
with t1:
    ctrl = st.radio("Controller to animate", nc.CONTROLLERS, index=3, horizontal=True,
                    format_func=lambda c: nc.LABEL[c])
    sim = simulate(scn, seed, ctrl, *sim_args[2:])
    s = sim.summary()
    c = st.columns(6)
    c[0].metric("Avg latency", f"{s['avg_latency']:.2f} ms")
    c[1].metric("P95 latency", f"{s['p95_latency']:.2f} ms")
    c[2].metric("Packet loss", f"{s['packet_loss']:.2f} %")
    c[3].metric("Throughput", f"{s['throughput']:.0f} Mbps")
    c[4].metric("Congested link-s", s["cong_link_seconds"])
    c[5].metric("Reroutes", s["reroutes"])
    st.caption("Press ▶ Play. Link colour and thickness = utilisation (green → red); hover the dots for numbers. "
               "Each direction of a cable is drawn separately.")
    pc(topology_figure(sim))
    a, b = st.columns(2)
    with a:
        f = go.Figure(go.Scatter(y=100 * sim.maxutil, line=dict(color=nc.COLOR[ctrl])))
        f.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="congestion (90%)")
        f.update_layout(title="Worst-link utilisation", xaxis_title="time (s)", yaxis_title="%", height=300,
                        margin=dict(t=40, b=10))
        pc(f)
    with b:
        names = [f"{x[0]}->{x[1]}" for x in sim.flows]
        hm = go.Figure(go.Heatmap(z=sim.place.T, y=names, x=list(range(nc.T)), showscale=True,
                                  colorscale=[[0, "#4c72b0"], [.5, "#dd8452"], [1, "#55a868"]],
                                  colorbar=dict(tickvals=[0, 1, 2], ticktext=nc.SPINES, title="spine")))
        hm.update_layout(title="Which spine carries each flow (changes = reroutes)", height=300,
                         margin=dict(t=40, b=10), xaxis_title="time (s)")
        pc(hm)
    st.subheader("Reroute log")
    if sim.events: dfs(pd.DataFrame(sim.events))
    else: st.info("This controller never rerouted traffic in this run.")

# ================================================================ TAB 2
with t2:
    sims = {k: simulate(scn, seed, k, *sim_args[2:]) for k in nc.CONTROLLERS}
    f = go.Figure()
    for k, sm in sims.items():
        f.add_trace(go.Scatter(y=100 * sm.maxutil, name=nc.LABEL[k], line=dict(color=nc.COLOR[k])))
    f.add_hline(y=90, line_dash="dash", line_color="red")
    f.update_layout(title="Worst-link utilisation — same traffic, four methods", xaxis_title="time (s)",
                    yaxis_title="%", height=350, margin=dict(t=40, b=10))
    pc(f)
    f = go.Figure()
    for k, sm in sims.items():
        f.add_trace(go.Scatter(y=sm.loss_t, name=nc.LABEL[k], line=dict(color=nc.COLOR[k])))
    f.update_layout(title="Packet loss over time (%)", xaxis_title="time (s)", height=280, margin=dict(t=40, b=10))
    pc(f)
    tab = pd.DataFrame([{"method": nc.LABEL[k], **{m: round(v, 3) for m, v in sm.summary().items()}}
                        for k, sm in sims.items()])
    dfs(tab)
    st.markdown("#### Statistical comparison over many seeds")
    n = st.slider("Number of seeds", 3, 20, 8)
    if st.button("Run multi-seed comparison for this scenario"):
        with st.spinner("Simulating..."):
            dfs(multi_seed(scn, n, model_name, p_thr, W, lag))
        st.caption("mean ± 95% confidence interval (t-distribution) over seeds; identical traffic for each method.")

# ================================================================ TAB 3
with t3:
    clf = mo["clf"]
    st.subheader("Held-out predictor performance (episodes never used for training)")
    dfs(pd.DataFrame([{"model": k, **{m: round(v, 3) for m, v in c.items() if m != "cm"}}
                      for k, c in clf.items()]))
    st.warning("Accuracy/F1 look excellent because most 'congested within 8 s' samples are links that are "
               "**already** congested. The honest early-warning metrics (columns `early_*`) only count links "
               "that were NOT yet congested — recall there is much lower.")
    a, b, c3 = st.columns(3)
    cm = np.array(clf[model_name]["cm"])
    with a:
        f = px.imshow(cm, text_auto=True, x=["No cong.", "Cong."], y=["No cong.", "Cong."],
                      color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual"),
                      title=f"Confusion matrix — {model_name}")
        f.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=50)); pc(f)
    y, pr = mo["test"]["y"], mo["test"]["p"]
    with b:
        f = go.Figure()
        for k, p in pr.items():
            fp, tp, _ = roc_curve(y[::5], p[::5]); f.add_trace(go.Scatter(x=fp, y=tp, name=k))
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="grey"), showlegend=False))
        f.update_layout(title="ROC curves", height=320, xaxis_title="FPR", yaxis_title="TPR", margin=dict(t=50)); pc(f)
    with c3:
        imp = pd.Series(mo["importances"]).sort_values()
        f = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#2a7ab0"))
        f.update_layout(title="Random Forest feature importance", height=320, margin=dict(t=50)); pc(f)

    st.subheader("Does it warn ahead? (traffic with NO rerouting)")
    P = unmanaged_probs(scn, seed, model_name)
    ecmp = simulate(scn, seed, "ecmp", model_name, p_thr, W, lag)
    cong = ecmp.true_util > nc.CONG
    leads, total = [], 0
    for l in range(ecmp.L):
        for t in np.where(cong[1:, l] & ~cong[:-1, l])[0] + 1:
            total += 1
            w = [u for u in range(max(0, t - 15), t) if P[u, l] >= p_thr]
            if w: leads.append(t - min(w))
    c = st.columns(3)
    c[0].metric("Congestion onsets", total)
    c[1].metric("Warned ahead (≤15 s before)", f"{len(leads)} / {total}")
    c[2].metric("Median warning lead", f"{np.median(leads):.0f} s" if leads else "—")
    link = st.selectbox("Link to inspect", [f"{u}->{v}" for u, v in ecmp.links],
                        index=int(np.argmax(ecmp.true_util.max(0))))
    li = [f"{u}->{v}" for u, v in ecmp.links].index(link)
    f = go.Figure()
    f.add_trace(go.Scatter(y=100 * ecmp.true_util[:, li], name="utilisation %", line=dict(color="#333")))
    f.add_trace(go.Scatter(y=100 * P[:, li], name=f"{model_name} P(congestion in {nc.H}s) ×100",
                           line=dict(color="#2a7ab0")))
    f.add_hline(y=90, line_dash="dash", line_color="red"); f.add_hline(y=100 * p_thr, line_dash="dot", line_color="#2a7ab0")
    f.update_layout(height=320, xaxis_title="time (s)", margin=dict(t=20)); pc(f)

# ================================================================ TAB 4
with t4:
    st.subheader("Congestion-aware Dijkstra, step by step")
    st.markdown("Pick a moment in the **unmanaged** run and ask: *if a new flow of rate r from X to Y appeared now, "
                "which path would Eq. (1) choose?* Our own heap-based Dijkstra answers; NetworkX double-checks it.")
    base = simulate(scn, seed, "ecmp", model_name, p_thr, W, lag)
    c = st.columns(4)
    src = c[0].selectbox("Source", [f"S{i}" for i in range(1, 9)], 0)
    dst = c[1].selectbox("Destination", [f"S{i}" for i in range(1, 9)], 5)
    tt = c[2].slider("Time (s)", 0, nc.T - 1, int(np.argmax(base.maxutil)))
    rate = c[3].slider("Flow rate (Mbps)", 50, 600, 250, 25)
    if src == dst:
        st.error("Choose different source and destination.")
    else:
        df, ours, theirs = nc.candidate_paths(base, src, dst, tt, rate, W)
        st.success(f"Chosen path: **{' → '.join(ours)}**   |   NetworkX agrees: **{ours == theirs or 'cost-equal'}**")
        show = df.copy()
        for k in ("util_term", "latency_term", "loss_term", "hop_term", "total"): show[k] = show[k].round(3)
        dfs(show)
        f = px.bar(df.melt(id_vars=["path"], value_vars=["util_term", "latency_term", "loss_term", "hop_term"]),
                   x="value", y="path", color="variable", orientation="h",
                   title="Cost of each candidate path, split into Eq. (1) terms",
                   labels=dict(value="weighted cost", variable="term"))
        f.update_layout(height=300, margin=dict(t=50)); pc(f)
        u = base.true_util[tt]
        u = base.true_util[tt]
        fab = [f"{a}→{b} {100 * u[i]:.0f}%" for i, (a, b) in enumerate(base.links)
               if a in nc.SPINES or b in nc.SPINES]
        st.caption("Leaf↔spine link utilisation at this moment: " + ", ".join(fab))
    with st.expander("Show our Dijkstra implementation (netcore.py)"):
        st.code(inspect.getsource(nc.dijkstra), language="python")

# ================================================================ TAB 5
with t5:
    R = nc.RES
    st.markdown("Produced by `python netcore.py` and `python ablation.py` (not re-run inside the app).")
    for fn, cap in (("fig_results.png", "Main results (mean ± 95% CI)"),
                    ("fig_timeseries.png", "Worst-link utilisation time series"),
                    ("fig_ablation.png", "Ablation / sensitivity")):
        if os.path.exists(os.path.join(R, fn)): st.markdown(f"**{cap}**"); img(os.path.join(R, fn))
    for fn, cap in (("paired_tests.csv", "NetCore − Reactive, paired tests"),
                    ("ablation_results.csv", "Ablation table"), ("results_ci.csv", "All metrics with 95% CI")):
        p = os.path.join(R, fn)
        if os.path.exists(p):
            with st.expander(cap): dfs(pd.read_csv(p))
    if not os.path.exists(os.path.join(R, "results_ci.csv")):
        st.info("No result files yet — run `python netcore.py` and `python ablation.py` first.")

# ================================================================ TAB 6
with t6:
    st.markdown(f"""
### Pipeline
`Monitor (1 Hz) → clean + features → ML predictor → P(congestion in next {nc.H}s) ≥ threshold?
→ congestion-aware Dijkstra (Eq. 1) → reroute the largest flow → performance analysis`

| Component | Where |
|---|---|
| Hand-written Dijkstra (binary heap) | `netcore.dijkstra()` |
| Eq. (1) cost: α·Util + β·Lat + γ·Loss + δ·Hops | `Sim.parts()`, `Sim.best_path()` |
| Predictors: Random Forest, Logistic Reg., Decision Tree, XGBoost | `netcore.train_all()` |
| Reactive baseline / ECMP / ideal spray | `Sim.control()`, `Sim.__init__()` |
| 95 % CIs and paired t-tests | `netcore.mean_ci()`, `netcore.paired_test()` |

### Baselines
* **ECMP (static hash)** – flows keep their hashed path, never rerouted.
* **Ideal spray** – every flow split equally over all spines; an *upper-bound reference* that real
  switches rarely achieve (reordering), shown for context.
* **Reactive** – same rerouting logic as NetCore but triggered only after measured utilisation > 90 %.

### Honest limitations
* Fluid-flow simulator: no TCP dynamics, packets or reroute transients.
* Training and test traffic come from the same synthetic generator (different seeds).
* The ML advantage over a well-tuned reactive controller is small here; see the *Experiments* tab for CIs.
""")
