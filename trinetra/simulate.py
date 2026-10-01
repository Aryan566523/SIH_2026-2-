"""Synthetic Automatic Weather Station network with physically consistent weather,
genuine extreme events and injected sensor / data faults (with ground-truth labels).

Real IMD archives can be dropped in instead (see cli `detect --csv`); the simulator exists
so that detection can be scored exactly, which is how the problem statement is evaluated
("anomaly injected data")."""
from dataclasses import dataclass, field
import zlib
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from .physics import sat_vp, rh_from_dew, station_pressure, RANGE

CITIES = [  # code, main station id, lon, lat, elev(m), Tmean, Tamp, Tdmean, name
    ("DEL", "AWS-DEL-014", 77.20, 28.60, 216, 34, 6.5, 12, "Delhi"),
    ("PUN", "AWS-PUN-003", 73.86, 18.52, 560, 29, 6.0, 17, "Pune"),
    ("CHN", "AWS-CHN-021", 80.27, 13.08, 10, 31, 3.5, 23, "Chennai"),
    ("MUM", "AWS-MUM-007", 72.88, 19.08, 14, 31, 2.5, 24, "Mumbai"),
    ("JPR", "AWS-JPR-011", 75.79, 26.91, 431, 34, 7.0, 11, "Jaipur"),
    ("HYD", "AWS-HYD-019", 78.48, 17.39, 542, 31, 5.5, 18, "Hyderabad"),
    ("KOL", "AWS-KOL-005", 88.36, 22.57, 6, 31, 3.5, 25, "Kolkata"),
    ("BLR", "AWS-BLR-012", 77.59, 12.97, 920, 26, 4.5, 17, "Bengaluru"),
    ("LEH", "AWS-LEH-001", 77.58, 34.16, 3500, 12, 9.0, -4, "Leh"),
]
ISOLATED = {"LEH"}          # a remote station with no neighbours inside 100 km


def make_network(seed=0):
    r = np.random.default_rng(seed)
    rows = []
    for code, sid, lon, lat, elev, *_ in CITIES:
        rows.append(dict(station=sid, city=code, lon=lon, lat=lat, elev=elev, main=True))
        if code in ISOLATED:
            continue
        for k in range(6):
            rows.append(dict(station=f"AWS-{code}-{30 + k * 7 + int(r.integers(0, 5)):03d}", city=code,
                             lon=lon + (r.random() - .5) * .9, lat=lat + (r.random() - .5) * .9,
                             elev=elev + float(r.normal(0, 25)), main=False))
    return pd.DataFrame(rows)


def _ar1(n, tau, sigma, r):
    phi = np.exp(-1.0 / tau)
    return lfilter([1.0], [1.0, -phi], sigma * np.sqrt(1 - phi ** 2) * r.standard_normal(n))


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _event_shape(kind, n, t0, r):
    """Return (dT, dTd, dP) arrays for a genuine meteorological event beginning at t0."""
    i = np.arange(n, dtype=float)
    dT = np.zeros(n); dTd = np.zeros(n); dP = np.zeros(n)
    if kind == "heatwave":
        H = r.uniform(6, 9); up, dur, down = 100, r.integers(240, 420), 100
        env = _smooth((i - t0) / up) * (1 - _smooth((i - t0 - up - dur) / down))
        dT, dTd = H * env, -2.5 * env
        dP = -1.2 * env
    elif kind == "thunderstorm":                      # gust front: cold pool + pressure surge
        D = r.uniform(5, 8); k = np.clip(i - t0, 0, None)
        drop = _smooth((i - t0) / 12.0)
        rec = np.exp(-np.clip(i - t0 - 12, 0, None) / 70.0)
        dT = -D * drop * rec
        dTd = 1.5 * drop * rec
        surge = 1.6 * _smooth((i - t0) / 5.0) * np.exp(-k / 25.0)
        dip = -1.4 * _smooth((i - t0 - 10) / 25.0) * np.exp(-np.clip(i - t0 - 35, 0, None) / 90.0)
        dP = surge + dip
    elif kind == "cold_front":
        env = _smooth((i - t0) / 60.0) * np.exp(-np.clip(i - t0 - 60, 0, None) / 360.0)
        dT, dTd, dP = -r.uniform(4, 6) * env, -4.0 * env, 2.5 * env
    elif kind == "fog":
        dur = r.integers(180, 300)
        env = _smooth((i - t0) / 60.0) * (1 - _smooth((i - t0 - dur) / 60.0))
        dT, dTd = -3.0 * env, 2.0 * env
    return dT, dTd, dP


@dataclass
class Sim:
    stations: pd.DataFrame
    time: pd.DatetimeIndex
    meas: dict                     # var -> (n_stn, n) measured values incl. NaN / sentinels
    truth: dict                    # var -> (n_stn, n) clean physical values
    anom: np.ndarray               # (n_stn, n) object array: '' or anomaly type
    anomalies: list = field(default_factory=list)
    events: list = field(default_factory=list)

    def to_long(self):
        n = len(self.time)
        out = []
        for i, s in enumerate(self.stations.itertuples()):
            out.append(pd.DataFrame({"station": s.station, "time": self.time, "T": self.meas["T"][i], "P": self.meas["P"][i],
                                     "RH": self.meas["RH"][i], "anomaly": self.anom[i]}))
        return pd.concat(out, ignore_index=True)


def simulate(stations, minutes, seed=1, start="2026-06-10 00:00", events=True, event_rate=0.7, force_events=()):
    """Simulate `minutes` of 1-minute observations. Returns a Sim with clean truth."""
    r = np.random.default_rng(seed)
    n = int(minutes)
    time = pd.date_range(start, periods=n, freq="min")
    hour = (time.hour.values + time.minute.values / 60.0)
    ns = len(stations)
    truth = {v: np.zeros((ns, n)) for v in ("T", "P", "RH")}
    ev_list = []
    for code, sid, lon, lat, elev, Tm, Ta, Tdm, name in CITIES:
        idx = np.where(stations.city.values == code)[0]
        if len(idx) == 0:
            continue
        # regional weather shared by the cluster
        Tc = Tm + Ta * np.cos(2 * np.pi * (hour - 14.5) / 24) + _ar1(n, 2 * 1440, 1.8, r) + _ar1(n, 25, .25, r)
        Tdc = Tdm - .8 * np.cos(2 * np.pi * (hour - 14) / 24) + _ar1(n, 1.5 * 1440, 1.4, r) + _ar1(n, 30, .12, r)
        dPc = 0.9 * np.cos(2 * np.pi * (hour - 10) / 12) + .3 * np.cos(2 * np.pi * (hour - 16) / 24) + _ar1(n, 2 * 1440, 1.8, r)
        dTe = np.zeros(n); dTde = np.zeros(n); dPe = np.zeros(n)
        if events and code not in ISOLATED:
            n_ev = r.poisson(event_rate * n / 1440.0)
            for _ in range(n_ev):
                kind = r.choice(["heatwave", "thunderstorm", "cold_front", "fog"], p=[.25, .35, .2, .2])
                t0 = int(r.integers(int(min(240, n // 4)), max(int(min(240, n // 4)) + 1, n - 120)))
                if kind == "fog":                       # fog forms at night
                    day0 = (t0 // 1440) * 1440
                    t0 = int(day0 + 1440 * (t0 % 1440 > 480) + r.integers(60, 240))
                    if t0 >= n - 60:
                        continue
                a, b, c = _event_shape(kind, n, t0, r)
                dTe += a; dTde += b; dPe += c
                ev_list.append(dict(city=code, kind=kind, start=t0, end=int(min(n, t0 + {"heatwave": 620, "thunderstorm": 150, "cold_front": 300, "fog": 300}[kind]))))
        for (fc, kind, t0) in force_events:            # scripted genuine events, used by the dashboard scenarios
            if fc in (code, "ALL") and code not in ISOLATED:
                a, b, c = _event_shape(kind, n, int(t0), r)
                dTe += a; dTde += b; dPe += c
                ev_list.append(dict(city=code, kind=kind, start=int(t0), end=int(min(n, t0 + 300))))
        Tc = Tc + dTe; Tdc = Tdc + dTde; dPc = dPc + dPe
        for j in idx:
            st = stations.iloc[j]
            # siting offsets belong to the station, so they are identical in every simulation run
            ro = np.random.default_rng(zlib.crc32(st.station.encode()))
            oT, oTd, oP = ro.normal(0, .3), ro.normal(0, .4), ro.normal(0, .15)
            Ti = Tc - 0.0065 * (st.elev - elev) + oT + _ar1(n, 30, .1, r)
            Tdi = Tdc + oTd + _ar1(n, 40, .08, r)
            Tdi = np.minimum(Tdi, Ti - 0.2)
            Pi = station_pressure(st.elev) + oP + dPc + _ar1(n, 60, .05, r)
            truth["T"][j], truth["P"][j], truth["RH"][j] = Ti, Pi, rh_from_dew(Ti, Tdi)
    meas = {v: truth[v].copy() for v in truth}
    meas["T"] += r.normal(0, .06, meas["T"].shape); meas["P"] += r.normal(0, .04, meas["P"].shape)
    meas["RH"] += r.normal(0, .4, meas["RH"].shape)
    meas["RH"] = np.clip(meas["RH"], 1, 100)
    meas["T"] = np.round(meas["T"], 2); meas["P"] = np.round(meas["P"], 2); meas["RH"] = np.round(meas["RH"], 1)
    return Sim(stations.reset_index(drop=True), time, meas, truth, np.full((ns, n), "", dtype=object), [], ev_list)


# ----------------------------------------------------------------------------- fault injection
KINDS = ["spike", "level_shift", "stuck", "drift", "noise_burst", "comm_error", "power_glitch"]
# Half of all injected faults are "gross" (obvious) and half "subtle" (near the accuracy tolerance),
# because real faults are often subtle and rule-based checks only catch the gross ones.
MAG = {"T": (6, 20), "P": (6, 22), "RH": (22, 45)}
MAG_SUBTLE = {"T": (1.5, 4), "P": (1.5, 4), "RH": (7, 15)}
DRIFT_MAG = {"T": (3, 8), "P": (3, 8), "RH": (10, 25)}
DRIFT_SUBTLE = {"T": (1.5, 3.5), "P": (1.5, 3.5), "RH": (6, 12)}
NOISE_SIG = {"T": (1.5, 4), "P": (1.5, 3), "RH": (6, 14)}
NOISE_SUBTLE = {"T": (.6, 1.2), "P": (.5, 1.0), "RH": (2.5, 5)}


def inject(sim, station_idx, kind, start, dur, var=None, r=None, sign=None, mag=None, form=None):
    """Inject one fault into a station (mutates sim.meas). Returns the ground-truth record."""
    r = r or np.random.default_rng(0)
    n = sim.meas["T"].shape[1]
    i0, i1 = int(start), int(min(n, start + dur))
    var = var or r.choice(["T", "P", "RH"], p=[.45, .3, .25])
    sg = sign if sign is not None else r.choice([-1, 1])
    m = sim.meas
    subtle = bool(r.random() < .5) if mag is None else False
    mg, dm, ns_ = (MAG_SUBTLE, DRIFT_SUBTLE, NOISE_SUBTLE) if subtle else (MAG, DRIFT_MAG, NOISE_SIG)
    rec = dict(station=sim.stations.station.iloc[station_idx], idx=int(station_idx), type=kind, var=str(var), start=i0, end=i1,
               strength="subtle" if (subtle and kind in ("spike", "level_shift", "drift", "noise_burst")) else "gross")
    seg = np.arange(i0, i1)
    if kind == "spike":
        i1 = min(n, i0 + int(r.integers(1, 5))); seg = np.arange(i0, i1); rec["end"] = i1
        a = mag or r.uniform(*mg[var]); m[var][station_idx, seg] += sg * a
    elif kind == "level_shift":
        a = mag or r.uniform(*mg[var])
        ramp = np.clip((seg - i0 + 1) / 2.0, 0, 1)
        m[var][station_idx, seg] += sg * a * ramp
    elif kind == "drift":
        a = mag or r.uniform(*dm[var])
        m[var][station_idx, seg] += sg * a * (seg - i0 + 1) / max(1, (i1 - i0))
    elif kind == "stuck":
        vs = ["T", "P", "RH"] if r.random() < .5 else [str(var)]
        rec["var"] = "+".join(vs)
        for v in vs:
            m[v][station_idx, seg] = m[v][station_idx, max(0, i0 - 1)]
    elif kind == "noise_burst":
        s = mag or r.uniform(*ns_[var]); m[var][station_idx, seg] += r.normal(0, s, len(seg))
    elif kind == "comm_error":
        if (form == "dropout") if form else (r.random() < .5):
            for v in ("T", "P", "RH"): m[v][station_idx, seg] = np.nan
            rec["var"] = "T+P+RH"; rec["form"] = "dropout"
        else:
            m["T"][station_idx, seg] = -99.9; m["P"][station_idx, seg] = 0.0; m["RH"][station_idx, seg] = -99.9
            rec["var"] = "T+P+RH"; rec["form"] = "sentinel"
    elif kind == "power_glitch":
        hit = seg[r.random(len(seg)) < .35]
        if len(hit) == 0: hit = seg[:1]
        for v in ("T", "P", "RH"):
            lo, hi = MAG[v]
            m[v][station_idx, hit] += r.choice([-1, 1], len(hit)) * r.uniform(lo, hi, len(hit)) * .8
        rec["var"] = "T+P+RH"
    sim.anom[station_idx, i0:rec["end"]] = kind
    rec["magnitude"] = float(a) * (sg if kind in ("spike", "level_shift", "drift") else 1) if kind in ("spike", "level_shift", "drift") else None
    sim.anomalies.append(rec)
    return rec


def inject_random(sim, per_station_day=0.9, seed=5, warmup=300, protect=None):
    """Randomly inject faults across the network (no two on one station within 90 min)."""
    r = np.random.default_rng(seed)
    n = sim.meas["T"].shape[1]
    for i in range(len(sim.stations)):
        k = r.poisson(per_station_day * (n - warmup) / 1440.0)
        t = warmup
        taken = []
        for _ in range(k):
            kind = r.choice(KINDS, p=[.14, .18, .16, .14, .12, .14, .12])
            dur = {"spike": 3, "level_shift": int(r.integers(60, 300)), "stuck": int(r.integers(30, 240)), "drift": int(r.integers(180, 600)),
                   "noise_burst": int(r.integers(20, 90)), "comm_error": int(r.integers(5, 60)), "power_glitch": int(r.integers(10, 40))}[kind]
            for _try in range(20):
                s = int(r.integers(warmup, max(warmup + 1, n - dur - 20)))
                if all(s > b + 90 or s + dur + 90 < a for a, b in taken):
                    inject(sim, i, kind, s, dur, r=r); taken.append((s, s + dur)); break
    return sim.anomalies


def inject_degradation(sim, station_idx, var, rate_per_day, start=0):
    """Slow sensor ageing (unlabelled): a bias that grows linearly. Used to test maintenance prediction."""
    n = sim.meas[var].shape[1]
    t = np.arange(n)
    ramp = np.clip(t - start, 0, None) / 1440.0 * rate_per_day
    if var == "RH":
        sim.meas["RH"][station_idx] = np.clip(sim.meas["RH"][station_idx] * (1 + ramp / 100.0), 1, 100)
    else:
        sim.meas[var][station_idx] += ramp
    rec = dict(station=sim.stations.station.iloc[station_idx], idx=int(station_idx), var=var, rate_per_day=rate_per_day)
    sim.degradations = getattr(sim, "degradations", []) + [rec]
    return rec
