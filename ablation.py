"""
Ablation, sensitivity and confidence-interval analysis for NetCore.

    python ablation.py              # 10 seeds x 5 scenarios per configuration
    python ablation.py --seeds 5    # faster
    python ablation.py --jobs -1    # use all CPU cores

Every configuration is run on the SAME traffic (same scenario + seed), so differences
against the default NetCore are tested with a PAIRED t-test and reported with a 95 % CI.
Outputs (results/): ablation_results.csv, ablation.md, fig_ablation.png
"""
import os, time, argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import netcore as nc
from netcore import SCENARIOS, METRICS, RES

KEY = ["cong_link_seconds", "p95_latency", "packet_loss", "reroutes"]


def flat(res, c):
    return {k: np.array([r[k] for sc in SCENARIOS for r in res[sc][c]]) for k in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True); t0 = time.time()
    seeds = list(range(4000, 4000 + a.seeds))        # different seeds from the main evaluation
    mo = nc.train_all()
    rf = mo["models"]["Random Forest"]
    ev = lambda model, c="netcore", **kw: flat(nc.evaluate(model, SCENARIOS, seeds, [c], a.jobs, **kw), c)

    print("default NetCore ..."); base = ev(rf)
    rows, store = [], {}

    def add(group, config, F, x=None):
        row = dict(group=group, config=config, x=x)
        for k in KEY + ["avg_latency", "cong_events"]:
            m, h = nc.mean_ci(F[k]); row[k] = m; row[k + "_ci"] = h
        for k in ("cong_link_seconds", "p95_latency", "packet_loss"):
            d, h, p = nc.paired_test(F[k], base[k])
            row[f"d_{k}"], row[f"d_{k}_ci"], row[f"p_{k}"] = d, h, p
        rows.append(row); store[(group, config)] = F
        print(f"  {group:10s} {config:34s} cong_link_s={row['cong_link_seconds']:7.1f}  "
              f"p95={row['p95_latency']:.2f}  reroutes={row['reroutes']:.1f}")

    add("reference", "ECMP (no rerouting)", ev(None, "ecmp"))
    add("reference", "Ideal spray", ev(None, "spray"))
    add("reference", "Reactive", ev(None, "reactive"))
    add("reference", "NetCore (default)", base)

    # 1. probability threshold ---------------------------------------------------------------
    print("threshold sweep")
    for thr in (0.2, 0.3, 0.4, 0.6, 0.7, 0.8):
        add("threshold", f"p >= {thr}", ev(rf, p_thr=thr), x=thr)
    rows.append({**[r for r in rows if r["config"] == "NetCore (default)"][0], "group": "threshold",
                 "config": "p >= 0.5", "x": 0.5})

    # 2. Eq.(1) cost-function weights (alpha, beta, gamma, delta) ----------------------------------
    print("cost-function weights")
    for name, w in {"util only (1,0,0,0)": (1, 0, 0, 0), "hops only (0,0,0,1)": (0, 0, 0, 1),
                    "no hop term (.5,.25,.25,0)": (.5, .25, .25, 0),
                    "latency heavy (.2,.6,.1,.1)": (.2, .6, .1, .1),
                    "loss heavy (.2,.1,.6,.1)": (.2, .1, .6, .1),
                    "equal (.25,.25,.25,.25)": (.25, .25, .25, .25)}.items():
        add("weights", name, ev(rf, weights=w))

    # 3. predictor family -------------------------------------------------------------------------
    print("predictor model")
    for name, m in mo["models"].items():
        if name != "Random Forest":
            add("model", name, ev(m))

    # 4. feature set (does short-term history help?) ---------------------------------------------------
    print("feature ablation + horizon (retraining)")
    Xtr, ytr, _, _ = nc.build_dataset(nc.train_ids())
    raw = nc.ColSel(RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=20,
                                           random_state=0), nc.RAW_COLS).fit(Xtr, ytr)
    raw.model.n_jobs = 1
    add("features", "raw 8 parameters only (no history)", ev(raw))

    # 5. prediction horizon ------------------------------------------------------------------------------
    for h in (4, 12):
        X, y, _, _ = nc.build_dataset(nc.train_ids(), horizon=h)
        m = RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=20,
                                   n_jobs=1, random_state=0).fit(X, y)
        add("horizon", f"label horizon = {h}s", ev(m), x=h)
    rows.append({**[r for r in rows if r["config"] == "NetCore (default)"][0], "group": "horizon",
                 "config": "label horizon = 8s", "x": 8})

    # 6. telemetry / control-loop delay -----------------------------------------------------------------
    print("telemetry lag")
    for lag in (0, 3, 6):
        add("lag", f"Reactive, lag={lag}s", ev(None, "reactive", lag=lag), x=lag)
        add("lag", f"NetCore, lag={lag}s", ev(rf, lag=lag), x=lag)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "ablation_results.csv"), index=False)
    write_md(df); make_fig(df)
    print(f"done in {time.time() - t0:.0f}s")


def write_md(df):
    L = ["# Ablation / sensitivity (5 scenarios x N seeds, paired against default NetCore)\n",
         "d_* = mean difference vs default NetCore (negative = better), [95% CI], paired t-test p.\n"]
    for g in ["reference", "threshold", "weights", "model", "features", "horizon", "lag"]:
        L += [f"\n## {g}\n", "| Config | Cong. link-s | P95 lat (ms) | Loss (%) | Reroutes | d cong-link-s [95% CI] | p |",
              "|---|---|---|---|---|---|---|"]
        for _, r in df[df.group == g].iterrows():
            L.append(f"| {r.config} | {r.cong_link_seconds:.1f} | {r.p95_latency:.2f} | {r.packet_loss:.3f} | "
                     f"{r.reroutes:.1f} | {r.d_cong_link_seconds:+.1f} [{r.d_cong_link_seconds - r.d_cong_link_seconds_ci:+.1f}, "
                     f"{r.d_cong_link_seconds + r.d_cong_link_seconds_ci:+.1f}] | {r.p_cong_link_seconds:.3f} |")
    open(os.path.join(RES, "ablation.md"), "w").write("\n".join(L))


def make_fig(df):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    t = df[df.group == "threshold"].sort_values("x")
    ax[0, 0].errorbar(t.x, t.d_cong_link_seconds, t.d_cong_link_seconds_ci, marker="o", color="#2a7ab0")
    ax[0, 0].set(title="(a) Probability threshold", xlabel="trigger threshold", ylabel="Δ cong. link-s vs default (paired CI)")
    b = ax[0, 0].twinx(); b.plot(t.x, t.reroutes, "s--", color="#e07b39"); b.set_ylabel("reroutes", color="#e07b39")

    def bar(a, g, title, extra=None):
        d = df[df.group == g]
        if extra is not None: d = pd.concat([d, extra])
        a.barh(d.config, d.d_cong_link_seconds, xerr=d.d_cong_link_seconds_ci, color="#2a7ab0", capsize=2)
        a.set(title=title, xlabel="Δ congested link-seconds vs default NetCore\n(paired 95% CI; lower = better)"); a.tick_params(axis="y", labelsize=7)
    default = df[df.config == "NetCore (default)"]
    bar(ax[0, 1], "weights", "(b) Eq.(1) weights", default)
    bar(ax[0, 2], "model", "(c) Predictor model", default.assign(config="Random Forest"))
    bar(ax[1, 0], "features", "(d) Feature set", default.assign(config="full 15 features (history)"))
    h = df[df.group == "horizon"].sort_values("x")
    ax[1, 1].errorbar(h.x, h.d_cong_link_seconds, h.d_cong_link_seconds_ci, marker="o", color="#2a7ab0")
    ax[1, 1].set(title="(e) Label horizon", xlabel="seconds ahead", ylabel="Δ cong. link-s vs default (paired CI)")
    for c, col in (("Reactive", "#e07b39"), ("NetCore", "#2a7ab0")):
        d = df[(df.group == "lag") & df.config.str.startswith(c)].sort_values("x")
        ax[1, 2].errorbar(d.x, d.d_cong_link_seconds, d.d_cong_link_seconds_ci, marker="o", label=c, color=col)
    ax[1, 2].set(title="(f) Telemetry / control delay", xlabel="lag (s)", ylabel="Δ cong. link-s vs default (paired CI)")
    ax[1, 2].legend()
    for a in ax.ravel(): a.axvline(0, color="k", lw=.5) if a.get_ylabel()=="" else a.axhline(0, color="k", lw=.5)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_ablation.png"), dpi=130); plt.close()


if __name__ == "__main__":
    main()
