"""Run the real engine on scripted scenarios and write data.js for the dashboard.

Every number the dashboard shows (series, alerts, confidence, explanations, corrected values, health)
comes from the engine, not from JavaScript."""
import json, os, time
import numpy as np
from . import physics as ph
from .engine import Panel, _nanmed_rows, _interp24, V
from .evaluate import train_model
from .incidents import Analyzer
from .health import sensor_health, maintenance_note
from .simulate import CITIES, make_network, simulate, inject, inject_degradation

START = "2026-06-10 06:00"
MINUTES = 600
WIN = 120
ONSET = 520          # minute of the scripted fault / event (window covers 14:00-16:00)

SCENARIOS = [
    ("level_shift", "Sudden jump", "fault"), ("spike", "Transient spike", "fault"), ("stuck", "Stuck value", "fault"),
    ("drift", "Slow drift", "fault"), ("noise_burst", "Noise burst", "fault"), ("comm_error", "Dropout", "fault"),
    ("power_glitch", "Power glitch", "fault"), ("storm", "Real storm", "event"), ("heatwave", "Real heatwave", "event"), ("clear", "Clear", "none"),
]
# which sensor is hit at each main station, and how hard (native units). Mirrors the slide: 55 C at Delhi.
VAR_OF = {"AWS-DEL-014": "T", "AWS-PUN-003": "P", "AWS-CHN-021": "T", "AWS-MUM-007": "T", "AWS-JPR-011": "P",
          "AWS-HYD-019": "RH", "AWS-KOL-005": "T", "AWS-BLR-012": "P", "AWS-LEH-001": "T"}
MAGS = {"level_shift": {"T": 17.0, "P": -9.0, "RH": 38.0}, "spike": {"T": 18.0, "P": 12.0, "RH": 40.0},
        "drift": {"T": 4.5, "P": 3.6, "RH": -13.0}, "noise_burst": {"T": 2.6, "P": 2.0, "RH": 9.0}}
DUR = {"level_shift": 200, "spike": 3, "stuck": 90, "drift": 160, "noise_burst": 80, "comm_error": 60, "power_glitch": 45}
DEFAULTS = {"AWS-DEL-014": "level_shift", "AWS-PUN-003": "drift", "AWS-CHN-021": "storm"}


def _round(a, d=2):
    a = np.asarray(a, float)
    return [None if not np.isfinite(x) else round(float(x), d) for x in a]


def _expected(m, ev, i, v):
    """What the sensor should read, from neighbours (None if there are none)."""
    idx = m.nb[i][0]
    if not idx:
        return None, None, None
    med, cnt = _nanmed_rows(ev["A"][v], idx)
    off = _interp24(m.off[v], ev["hour"]).T[i]
    cl = _interp24(m.clim_tab[v][i:i + 1], ev["hour"])[:, 0]
    sg = 3 * m.sig[v][i]
    if v == "RH":
        e = np.minimum(100, cl * np.exp((med + off) / 100.0))
        return e, np.minimum(100, e * np.exp(-sg / 100.0)), np.minimum(100, e * np.exp(sg / 100.0))
    e = cl + med + off
    return e, e - sg, e + sg


def run_scenario(model, stations, key, seed=900):
    sim = simulate(stations, MINUTES, seed=seed, start=START, events=False,
                   force_events=[("ALL", "thunderstorm", ONSET)] if key == "storm" else [("ALL", "heatwave", 470)] if key == "heatwave" else [])
    r = np.random.default_rng(seed + 1)
    if key in DUR:
        for j, s in enumerate(stations.station):
            if s not in VAR_OF:
                continue
            v = VAR_OF[s]
            start = 440 if key == "drift" else ONSET + (j % 5)
            kw = dict(var=v, r=np.random.default_rng(seed + j))
            if key in MAGS:
                m = MAGS[key][v]
                kw.update(sign=1 if m > 0 else -1, mag=abs(m))
            if key == "comm_error":
                kw["form"] = "dropout" if j % 2 == 0 else "sentinel"
            inject(sim, j, key, start, DUR[key], **kw)
    P = Panel.from_sim(sim)
    ev = model.score(P)
    A = Analyzer(model)
    incs = A.analyse(P, ev)
    return sim, P, ev, A, incs


def build_scenarios(model, stations):
    out = {}
    mains = [s for s in stations.station if s in VAR_OF]
    t0 = MINUTES - WIN
    for key, label, kind in SCENARIOS:
        sim, P, ev, A, incs = run_scenario(model, stations, key)
        for s in mains:
            i = int(stations.index[stations.station == s][0])
            mine = [x for x in incs if x["idx"] == i and x["detect"] >= t0 - 60 and x["end"] > t0]
            fault = sorted([x for x in mine if x["kind"] == "fault"], key=lambda x: -x["confidence"])
            event = [x for x in mine if x["kind"] == "event"]
            series, expd = {}, {}
            for v in V:
                raw = ev["raw"][v][i, t0:].copy()
                bad = ~np.isfinite(raw) | (raw <= -90) | (raw < ph.RANGE[v][0] - 100) | ((v == "P") & (raw <= 1))
                raw[bad] = np.nan
                series[v] = _round(raw, 2 if v != "RH" else 1)
                e, lo, hi = _expected(model, ev, i, v)
                expd[v] = None if e is None else dict(med=_round(e[t0:]), lo=_round(lo[t0:]), hi=_round(hi[t0:]))
            rec = dict(series=series, expected=expd, status="ok", incident=None, event=None,
                       p=_round(ev["p"][i, t0:], 3), ts=_round(np.minimum(ev["ts"][i, t0:], 5), 2))
            shift = lambda d: {**d, "onset": int(d["onset"] - t0), "detect": int(d["detect"] - t0), "end": int(d["end"] - t0),
                               "onset_clock": d["onset_time"][11:16], "detect_clock": d["detect_time"][11:16]}
            if fault:
                rec["status"] = "fault"; rec["incident"] = shift(fault[0])
            elif event:
                rec["status"] = "event"; rec["event"] = shift(event[0])
            out.setdefault(s, {})[key] = rec
        print("scenario", key, flush=True)
    return out


def health_block(model, stations):
    sim = simulate(stations, 1440 * 3, seed=77)
    ix = lambda s: int(stations.index[stations.station == s][0])
    for s, v, rate in [("AWS-PUN-003", "P", 0.3), ("AWS-DEL-014", "T", 0.35), ("AWS-KOL-005", "RH", -1.2)]:
        inject_degradation(sim, ix(s), v, rate)
    P = Panel.from_sim(sim); ev = model.score(P); A = Analyzer(model); inc = A.analyse(P, ev)
    H = sensor_health(model, ev, P, inc, A)
    return {s: dict(H[s], notes=maintenance_note(H[s])) for s in stations.station if s in VAR_OF}


def export(model=None, stations=None, outdir=".", results="results/evaluation.json"):
    t0 = time.time()
    if model is None:
        model, stations = train_model(0)
    net = []
    for s in stations.itertuples():
        if s.station not in VAR_OF:
            continue
        i = int(stations.index[stations.station == s.station][0])
        idx, km = model.nb[i]
        city = next(c for c in CITIES if c[0] == s.city)
        net.append(dict(id=s.station, city=s.city, name=city[8], lon=round(float(s.lon), 3), lat=round(float(s.lat), 3), elev=int(s.elev),
                        neighbours=[dict(id=stations.station.iloc[j], lon=round(float(stations.lon.iloc[j]), 3), lat=round(float(stations.lat.iloc[j]), 3), km=round(k, 1)) for j, k in zip(idx, km)]))
    data = dict(generated=time.strftime("%Y-%m-%d %H:%M"), window=dict(minutes=WIN, start="14:00", end="15:59"),
                scenarios=[dict(key=k, label=l, kind=kd) for k, l, kd in SCENARIOS], defaults=DEFAULTS, stations=net,
                runs=build_scenarios(model, stations), health=health_block(model, stations), units=ph.UNITS, names=ph.NAMES)
    if os.path.exists(results):
        data["evaluation"] = json.load(open(results))
    txt = "window.TRINETRA_DATA = " + json.dumps(data, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o)) + ";\n"
    with open(os.path.join(outdir, "data.js"), "w") as f:
        f.write(txt)
    print(f"data.js {len(txt) / 1e6:.2f} MB in {time.time() - t0:.0f}s")
    return data
