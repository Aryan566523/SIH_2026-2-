"""Run the complete evaluation and write results/evaluation.json (used by the website and the document)."""
import json, os, time
import numpy as np
from .evaluate import train_model, run_evaluation, WARMUP
from .engine import Panel
from .simulate import simulate, inject_random, inject_degradation
from .incidents import Analyzer
from .health import sensor_health, maintenance_note
from .stream import StreamMonitor


def _clean(r):
    return {k: v for k, v in r.items()}


def build_report(outdir="results", model=None, stations=None, seeds=(101, 202, 303), stream_ticks=120, verbose=True):
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    if model is None:
        model, stations = train_model(0)
    log(f"trained in {time.time() - t0:.0f}s")
    rep = dict(generated=time.strftime("%Y-%m-%d %H:%M"), stations=int(len(stations)), test_days=4, seeds=list(seeds), runs=[])
    for sd in seeds:
        r = run_evaluation(model=model, test_seed=sd)
        rep["runs"].append(dict(seed=sd, rows=r["rows"], row_level=r["row_level"]))
        log(f"seed {sd}: full F1 {r['rows'][-1]['f1']}")
    # aggregate across seeds
    agg = {}
    for name in [x["name"] for x in rep["runs"][0]["rows"]]:
        rows = [next(x for x in run["rows"] if x["name"] == name) for run in rep["runs"]]
        keys = ["recall", "precision", "f1", "false_alarms_per_station_day", "median_latency_min", "p90_latency_min", "genuine_kept_rate", "root_cause_accuracy"]
        agg[name] = {k: (round(float(np.mean([x[k] for x in rows if x[k] is not None])), 3) if any(x[k] is not None for x in rows) else None) for k in keys}
        agg[name]["injected"] = int(sum(x["injected"] for x in rows)); agg[name]["detected"] = int(sum(x["detected"] for x in rows))
        agg[name]["false_alarms"] = int(sum(x["false_alarms"] for x in rows)); agg[name]["alerts"] = int(sum(x["alerts"] for x in rows))
    full_rows = [next(x for x in run["rows"] if x["name"] == "TRINETRA (full)") for run in rep["runs"]]
    per = {}
    for x in full_rows:
        for t, d in x["per_type"].items():
            a = per.setdefault(t, [0, 0]); a[0] += d["n"]; a[1] += round(d["recall"] * d["n"])
    agg["per_type"] = {t: dict(n=a[0], recall=round(a[1] / a[0], 3)) for t, a in per.items()}
    rep["summary"] = agg
    # stress tests
    log("stress tests")
    st1 = run_evaluation(model=model, test_seed=404, no_neighbours=True)
    st2 = run_evaluation(model=model, test_seed=505, rate=2.5)
    pick = lambda r: next(x for x in r["rows"] if x["name"] == "TRINETRA (full)")
    keep = ["recall", "precision", "f1", "false_alarms_per_station_day", "median_latency_min", "genuine_kept_rate", "root_cause_accuracy", "injected"]
    rep["stress"] = {"No neighbours (physics + AI only)": {k: pick(st1)[k] for k in keep}, "Dense faults (2.5 per station-day)": {k: pick(st2)[k] for k in keep}}
    rep["stress_baseline"] = {"No neighbours (physics + AI only)": {k: st1["rows"][0][k] for k in keep}, "Dense faults (2.5 per station-day)": {k: st2["rows"][0][k] for k in keep}}
    # maintenance prediction
    log("health test")
    sim = simulate(stations, 1440 * 3, seed=77)
    ix = lambda s: int(stations.index[stations.station == s][0])
    plan = [("AWS-PUN-003", "P", 0.3), ("AWS-DEL-014", "T", 0.35), ("AWS-KOL-005", "RH", -1.2)]
    for s, v, rate in plan:
        inject_degradation(sim, ix(s), v, rate)
    P = Panel.from_sim(sim); ev = model.score(P); A = Analyzer(model); inc = A.analyse(P, ev)
    H = sensor_health(model, ev, P, inc, A)
    from .physics import TOLERANCE
    rows = []
    for s, v, rate in plan:
        h = H[s][v]
        rows.append(dict(station=s, var=v, true_rate=rate, est_rate=h["drift_per_day"], status=h["status"], bias=h["bias"], days_to_limit=h["days_to_limit"]))
    rep["health_test"] = rows
    # streaming
    log("streaming test")
    te = simulate(stations, 330, seed=88); inject_random(te, seed=89, per_station_day=9, warmup=100)
    Pt = Panel.from_sim(te); batch_inc = [x for x in Analyzer(model).analyse(Pt, model.score(Pt), with_events=False) if x["detect"] >= 100 + 150]
    sm = StreamMonitor(model, te.stations)
    got = []
    for k in range(len(te.time)):
        obs = {s: (te.meas["T"][j, k], te.meas["P"][j, k], te.meas["RH"][j, k]) for j, s in enumerate(te.stations.station)}
        for a in sm.push(te.time[k], obs):
            got.append((a["station"], a["detect"] + (k + 1 - len(sm.times))))
        if k >= 150 + stream_ticks - 1 + 100 and False:
            break
    bset = {(x["station"], x["detect"]) for x in batch_inc}
    gset = {g for g in got if g[1] >= 250}
    rep["streaming"] = dict(stations=int(len(stations)), window_min=sm.W, ms_median=round(float(np.median(sm.latency_ms)), 1), ms_p95=round(float(np.percentile(sm.latency_ms, 95)), 1),
                            ms_max=round(float(np.max(sm.latency_ms)), 1), batch_alerts=len(bset), stream_alerts=len(gset), identical=len(bset & gset))
    os.makedirs(outdir, exist_ok=True)
    json.dump(rep, open(os.path.join(outdir, "evaluation.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"done in {time.time() - t0:.0f}s")
    return rep
