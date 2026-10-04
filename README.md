# NetCore — AI-Based Predictive Network Congestion Management in Data Centers

A complete, runnable implementation of the NetCore paper: a leaf-spine data-center simulator, a
machine-learning congestion predictor, a **hand-written congestion-aware Dijkstra**, reactive / ECMP
baselines, statistical evaluation, and an interactive **Streamlit dashboard**.

```
monitor (1 Hz) → clean + features → ML predictor → P(congestion in next 8 s) ≥ threshold ?
        → congestion-aware Dijkstra (Eq. 1) → reroute largest flow → performance analysis
```

## Files

| File | Purpose |
|---|---|
| `netcore.py` | Core engine: topology, traffic, simulator, controllers, **Dijkstra**, ML training, evaluation, CIs |
| `app.py` | Streamlit dashboard (live animated topology, method comparison, predictor, Dijkstra explorer) |
| `ablation.py` | Ablation / sensitivity study with paired tests and 95 % confidence intervals |
| `test_dijkstra.py` | Unit tests: our Dijkstra == NetworkX on 300 random graphs + inside the simulator |
| `requirements.txt` | Python dependencies |
| `results/` | Generated tables (`*.csv`, `*.md`) and figures (`*.png`) |
| `cache/models.joblib` | Trained predictors (created automatically; delete to retrain) |

## Setup (macOS / Linux / Windows)

```bash
cd netcore_project
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **macOS + XGBoost:** if you see an error about `libomp`, run `brew install libomp`.
> If XGBoost still cannot load, the code automatically falls back to scikit-learn's gradient boosting
> (the model is then labelled "GradBoost (XGBoost unavailable)") — nothing else breaks.

## Run order

```bash
python test_dijkstra.py            # 1. verify the Dijkstra implementation       (1 s)
python netcore.py                  # 2. train models + main experiment, 20 seeds (~3-5 min)
python ablation.py                 # 3. ablations + sensitivity, 10 seeds        (~15-25 min)
streamlit run app.py               # 4. open the dashboard in your browser
```

Useful options: `python netcore.py --seeds 10 --jobs -1` (all CPU cores), `python netcore.py --retrain`,
`python ablation.py --seeds 5` (faster).

## What was added compared with the first version

1. **Own Dijkstra** (`netcore.dijkstra`, binary heap) — verified against NetworkX by `test_dijkstra.py`.
2. **Streamlit dashboard** with animated topology, reroute log, predictor diagnostics and a step-by-step
   Dijkstra explorer.
3. **Ablation + confidence intervals**: threshold, Eq. (1) weights, predictor family, feature set,
   prediction horizon, telemetry delay; paired t-tests against default NetCore.
4. **Baselines & models:** ECMP (static hash), ideal per-packet spray (upper-bound reference),
   reactive threshold, and **XGBoost** next to Random Forest / Logistic Regression / Decision Tree.

## Methods in one paragraph

Eight servers, two leaf and three spine switches (1 Gbps fabric links). Sixteen flows follow five
traffic scenarios (steady ramp, sudden burst, single hotspot, repeated bursts, multi-link). Every
second each link reports utilisation, traffic, packet rate, latency, loss, throughput, queue length and
flow count; 15 features (current values + 5-s means/trends/max) feed the predictor, which estimates the
probability that the link exceeds 90 % utilisation within 8 s. Links with probability ≥ threshold trigger
one reroute per link: the largest flow is re-placed on the path with minimum
`Cost = α·Util + β·Latency + γ·Loss + δ·Hops` (found by Dijkstra), only if the new bottleneck is at
least 5 points better. The **reactive** baseline uses identical rerouting logic but triggers only after
measured utilisation exceeds 90 %. All methods see identical traffic (same scenario + seed).

## Honest limitations

* Fluid-flow simulator: no TCP, no packets, no reroute transients; synthetic traffic.
* Training/test traffic come from the same generator (different seeds) — real traffic may differ.
* Headline classification scores are inflated by "already congested" samples; use the
  `early_*` columns for genuine early-warning ability.
* Ideal spray is an upper-bound reference (real switches reorder packets); it is not a fair competitor.
