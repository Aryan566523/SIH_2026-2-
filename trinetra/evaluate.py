"""Score detection on anomaly-injected data (the evaluation method in the problem statement)."""
import numpy as np
import collections
from sklearn.metrics import average_precision_score, roc_auc_score
from .engine import Trinetra, Panel
from .engine import P_THR
from .incidents import Analyzer, detect_runs
from .simulate import make_network, simulate, inject_random, KINDS

WARMUP = 300


def train_model(seed=0, train_days=8, val_days=3, verbose=False):
    st = make_network(seed)
    tr = simulate(st, 1440 * train_days, seed=seed + 11)
    m = Trinetra(seed=seed).fit(Panel.from_sim(tr))
    va = simulate(st, 1440 * val_days, seed=seed + 22)
    inject_random(va, seed=seed + 23, per_station_day=1.2)
    m.calibrate(Panel.from_sim(va), va.anom, warmup=WARMUP)
    return m, st


def _match(gt, incs, slack=20):
    """Match incidents to ground-truth anomalies on the same station."""
    by = collections.defaultdict(list)
    for k, x in enumerate(incs):
        by[x["idx"]].append(k)
    det, used = {}, set()
    for gi, g in enumerate(gt):
        best = None
        for k in by[g["idx"]]:
            x = incs[k]
            if x["onset"] <= g["end"] + slack and x["end"] >= g["start"]:
                ov = min(x["end"], g["end"] + slack) - max(x["onset"], g["start"])
                if best is None or ov > best[0]:
                    best = (ov, k)
        if best:
            det[gi] = best[1]; used.add(best[1])
    # any incident that overlaps a GT anomaly (even if a second incident) is not a false alarm
    for k, x in enumerate(incs):
        if k in used: continue
        for g in gt:
            if g["idx"] == x["idx"] and x["onset"] <= g["end"] + slack and x["end"] >= g["start"]:
                used.add(k); break
    return det, used


def score_incidents(name, incs, sim, days_scored):
    gt = [g for g in sim.anomalies if g["start"] >= WARMUP]
    faults = [x for x in incs if x["kind"] == "fault" and x["detect"] >= WARMUP]
    det, used = _match(gt, faults)
    fa = [k for k in range(len(faults)) if k not in used]
    lat = [max(0, faults[det[gi]]["detect"] - g["start"]) for gi, g in enumerate(gt) if gi in det]
    ok = [(gt[gi]["type"], faults[k]["type"]) for gi, k in det.items() if "type" in faults[k]]
    # genuine events wrongly reported as faults (a false alarm on any station of that city)
    st_city = sim.stations.city.values
    ev_hit = 0; n_ev = 0
    for e in sim.events:
        if e["start"] < WARMUP: continue
        n_ev += 1
        if any(faults[k]["city"] == e["city"] and faults[k]["onset"] < e["end"] and faults[k]["end"] > e["start"] for k in fa):
            ev_hit += 1
    per_type = {}
    for t in KINDS:
        g = [gi for gi, x in enumerate(gt) if x["type"] == t]
        if g:
            per_type[t] = dict(n=len(g), recall=round(sum(1 for gi in g if gi in det) / len(g), 3))
    tp = len(det); prec = (len(used)) / max(1, len(faults))
    rec = tp / max(1, len(gt))
    return dict(name=name, injected=len(gt), detected=tp, alerts=len(faults), false_alarms=len(fa),
                recall=round(rec, 3), precision=round(prec, 3), f1=round(2 * prec * rec / max(1e-9, prec + rec), 3),
                false_alarms_per_station_day=round(len(fa) / (len(sim.stations) * days_scored), 3),
                median_latency_min=float(np.median(lat)) if lat else None, p90_latency_min=float(np.percentile(lat, 90)) if lat else None,
                genuine_events=n_ev, genuine_events_flagged=ev_hit,
                genuine_kept_rate=round(1 - ev_hit / max(1, n_ev), 3),
                root_cause_accuracy=round(sum(a == b for a, b in ok) / max(1, len(ok)), 3) if ok else None,
                per_type=per_type, confusion=dict(collections.Counter(f"{a}->{b}" for a, b in ok if a != b)))


def baselines(ev, panel):
    """Threshold-only QC (range/step/stuck/missing) and temporal-AI-only (no physics/neighbour context)."""
    ns, n = ev["p"].shape
    out = {}
    qc = ev["hard"] | ev["step_any"]
    inc = []
    for i in range(ns):
        for (o, d, e, og) in detect_runs(qc[i].astype(float), thr=.5, need=1, win=1, close=10):
            inc.append(dict(kind="fault", idx=i, city=panel.stations.city.iloc[i], onset=o, detect=d, end=e, type="?"))
    out["Threshold QC only"] = inc
    inc = []
    for i in range(ns):
        for (o, d, e, og) in detect_runs((ev["ts"][i] >= 1.0).astype(float), thr=.5, need=4, win=6, close=10):
            inc.append(dict(kind="fault", idx=i, city=panel.stations.city.iloc[i], onset=o, detect=d, end=e, type="?"))
    out["Temporal AI only"] = inc
    return out


def run_evaluation(model=None, seed=0, test_days=4, test_seed=None, rate=0.9, no_neighbours=False, verbose=True):
    if model is None:
        model, st = train_model(seed)
    st = make_network(seed)
    te = simulate(st, 1440 * test_days, seed=(test_seed or seed + 33))
    inject_random(te, seed=(test_seed or seed + 33) + 1, per_station_day=rate, warmup=WARMUP)
    P = Panel.from_sim(te)
    saved = model.nb
    if no_neighbours:
        model.nb = [([], [])] * len(st)
    try:
        ev = model.score(P)
    finally:
        pass
    A = Analyzer(model)
    incs = A.analyse(P, ev)
    days = (test_days * 1440 - WARMUP) / 1440.0
    rows = []
    b = baselines(ev, P)
    for k, v in b.items():
        rows.append(score_incidents(k, v, te, days))
    full = score_incidents("TRINETRA (full)", incs, te, days)
    rows.append(full)
    y = (te.anom != "")[:, WARMUP:].ravel()
    row_level = dict(auc=float(roc_auc_score(y, ev["p"][:, WARMUP:].ravel())), ap=float(average_precision_score(y, ev["p"][:, WARMUP:].ravel())))
    model.nb = saved
    return dict(rows=rows, row_level=row_level, model=model, sim=te, panel=P, ev=ev, incidents=incs, days=days)
