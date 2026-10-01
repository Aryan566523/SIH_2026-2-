"""Turn per-minute evidence into alerts: incidents, root cause, confidence, explanation,
corrected values and repair advice."""
import numpy as np
from . import physics as ph
from .engine import P_THR, _nanmed_rows, _interp24

V = ph.VARS
CAUSE = {
    "spike": "Electrical interference or a transient sensor fault",
    "level_shift": "Sensor damage or a sudden calibration shift",
    "drift": "Calibration drift or sensor ageing",
    "stuck": "Frozen sensor or a logger that has hung",
    "noise_burst": "Loose connection or moisture in the sensor",
    "comm_error": "Communication, logger or power-supply failure",
    "power_glitch": "Power fluctuation (battery, solar controller or brown-out)",
}
TYPE_LABEL = {"spike": "Transient spike", "level_shift": "Sudden jump", "drift": "Slow drift", "stuck": "Stuck reading",
              "noise_burst": "Noise burst", "comm_error": "Communication error", "power_glitch": "Power glitch"}
ACTION = {
    "spike": "Inspect wiring and shielding of the {s} within 48 hours",
    "level_shift": "Replace or recalibrate the {s} within 48 hours",
    "drift": "Recalibrate the {s} within 7 days",
    "stuck": "Check the logger and wiring of the {s}, power-cycle and verify within 24 hours",
    "noise_burst": "Check connectors and moisture ingress on the {s} within 72 hours",
    "comm_error": "Check the GPRS/modem link, logger and power supply within 24 hours",
    "power_glitch": "Check the battery, solar charge controller and power supply within 48 hours",
}
SEV_ORDER = ["low", "medium", "high"]


def _f(x, d=1):
    return ("−" if x < 0 else "") + f"{abs(x):.{d}f}"


def _sg(x, d=1):
    return ("−" if x < 0 else "+") + f"{abs(x):.{d}f}"


def _runs(mask, gap):
    """Group True positions into (start, last) runs, merging gaps shorter than `gap`."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    out, s, l = [], idx[0], idx[0]
    for j in idx[1:]:
        if j - l > gap:
            out.append((s, l)); s = j
        l = j
    out.append((s, l))
    return out


def detect_runs(p, thr=P_THR, need=4, win=6, close=10, instant=None):
    """Alert state machine. Returns list of (onset, detect, end, ongoing).
    An alert opens when `need` of the last `win` minutes are above `thr`, or at once on an `instant` row."""
    n = len(p)
    act = p >= thr
    if instant is not None:
        act = act | instant
    cs = np.concatenate([[0], np.cumsum(act)])
    sm = np.array([cs[t + 1] - cs[max(0, t + 1 - win)] for t in range(n)])
    if instant is not None:
        sm = np.where(instant, np.maximum(sm, need), sm)
    out, t = [], 0
    while t < n:
        if sm[t] >= need:
            detect = t
            w = max(0, t - win + 1)
            onset = w + int(np.argmax(act[w:t + 1]))
            last = t
            while t < n and t - last < close:
                t += 1
                if t < n and act[t]:
                    last = t
            ongoing = t >= n and (n - 1 - last) < close
            out.append((onset, detect, min(n, last + 1), ongoing))
            t = last + close
        else:
            t += 1
    return out


class Analyzer:
    def __init__(self, model):
        self.m = model

    # ----------------------------------------------------------------- per station
    def _dev_native(self, ev, i, v):
        """Deviation of station i from neighbours in native units (or None without neighbours)."""
        if not ev["has_nb"][i].any():
            return None
        d = ev["zL"][v][i] * self.m.sig[v][i]
        if v == "RH":                                   # spatial humidity is a log-ratio; convert to %RH points
            d = ev["clean"]["RH"][i] * (1 - np.exp(-np.nan_to_num(d, nan=0.0) / 100.0))
        return d

    def analyse(self, panel, ev, stations=None, with_events=True):
        m = self.m
        ns, n = ev["p"].shape
        out = []
        for i in range(ns):
            if stations is not None and i not in stations:
                continue
            runs = detect_runs(ev["p"][i], instant=ev.get("instant", [None] * (i + 1))[i] if "instant" in ev else None)
            fault_mask = np.zeros(n, bool)
            for (o, d, e, og) in runs:
                inc = self._fault(panel, ev, i, o, d, e, og)
                out.append(inc)
                fault_mask[o:e] = True
            if with_events:
                out += self._events(panel, ev, i, fault_mask)
                out += self._extreme(panel, ev, i, fault_mask)
        return out

    # ----------------------------------------------------------------- fault incident
    def _fault(self, panel, ev, i, onset, detect, end, ongoing):
        m = self.m
        n = ev["p"].shape[1]
        seg = slice(onset, max(end, onset + 1))
        raw, clean = ev["raw"], ev["clean"]
        has_nb = bool(ev["has_nb"][i, seg].any())
        nb_idx, nb_km = m.nb[i]
        comm_frac = float(ev["comm"][i, seg].mean())
        stuck_v = [v for v in V if ev["stuck"][v][i, seg].any()]
        # ---- deviation series in native units
        dev, main, peak = {}, None, {}
        for v in V:
            d = self._dev_native(ev, i, v)
            if d is None:                               # no neighbours: deviation from own pre-fault level
                base_lo = max(0, onset - 90); base_hi = max(1, onset - 15)
                ref = np.nanmedian(clean[v][i, base_lo:base_hi]) if base_hi > base_lo else np.nan
                cl = _interp24(m.clim_tab[v][i:i + 1], ev["hour"])[:, 0]
                d = clean[v][i] - ref - (cl - cl[max(0, onset - 50)])
            if ev["rng"][v][i].any():                      # value outside the physical range was masked; measure it against the estimate
                ex = self._expected_series(ev, i, v)
                if ex is not None:
                    d = np.where(ev["rng"][v][i], raw[v][i] - ex, d)
            dev[v] = d
            with np.errstate(all="ignore"):
                peak[v] = float(np.nanmax(np.abs(d[seg]))) if np.isfinite(d[seg]).any() else 0.0
        zpeak = {v: (float(np.nanmax(np.abs(ev["zL"][v][i, seg]))) if has_nb and np.isfinite(ev["zL"][v][i, seg]).any() else 0.0) for v in V}
        for v in V:                                        # range violations are the strongest possible evidence
            if has_nb and ev["rng"][v][i, seg].any():
                zpeak[v] = max(zpeak[v], peak[v] / max(m.sig[v][i], 1e-6))
        # main variable: largest spatial z, else largest normalised deviation
        key = (lambda v: zpeak[v]) if has_nb else (lambda v: peak[v] / max(ph.TOLERANCE[v], 1e-6))
        main = max(V, key=key)
        # ---- classification
        elev = {v: float(np.nanmean(ev["F"][v]["dsd"].T[i, seg] > 4.0)) for v in V}
        nsp = {v: int(((np.abs(np.nan_to_num(ev["F"][v]["d1"].T[i, seg])) > ph.STEP[v] * .5)).sum()) for v in V}
        dev_rows = {v: int((np.abs(np.nan_to_num(dev[v][seg])) > 4 * max(ph.NOISE[v], .1) * (5 if v != "RH" else 2)).sum()) for v in V}
        typ = None
        n_act = int((ev["p"][i, seg] >= P_THR).sum())
        if comm_frac >= .5:
            typ = "comm_error"
        elif stuck_v and float(ev["stuck_any"][i, seg].mean()) >= .5:
            typ = "stuck"
        else:
            n_elev = sum(1 for v in V if elev[v] >= .35)
            n_step = sum(1 for v in V if nsp[v] >= 2)
            length = end - onset
            spiky = n_step >= 2 and n_elev >= 2
            # big minute-to-minute jumps in the main variable, and whether the deviation persists after the first minutes
            d1 = np.abs(np.nan_to_num(ev["F"][main]["d1"].T[i, seg]))
            k_big = int((d1 > 4 * 1.414 * ph.NOISE[main]).sum())
            late = slice(min(length - 1, max(5, length // 3)), length)
            if has_nb:
                zz = np.abs(np.nan_to_num(ev["zL"][main][i, seg]))
                zz = np.where(ev["rng"][main][i, seg], 99.0, zz)
                persist = float(np.median(zz[late])) >= 3.0 if length > 5 else (float(zz[-1]) >= 3.0 if ongoing else False)
            else:
                zz = np.abs(np.nan_to_num(dev[main][seg])) / max(ph.TOLERANCE[main], .1)
                persist = float(np.median(zz[late])) >= 1.0 if length > 5 else (float(zz[-1]) >= 1.0 if ongoing else False)
            dv = np.nan_to_num(dev[main][seg])
            big = np.abs(dv) > 3 * max(ph.NOISE[main] * 4, .5 * ph.TOLERANCE[main])
            sgn = max(float((dv[big] > 0).mean()), float((dv[big] < 0).mean())) if big.sum() >= 4 else 1.0
            if spiky and comm_frac < .5:
                typ = "power_glitch"
            elif k_big / max(1, length) >= .25 and k_big > 8 and sgn < .85:
                typ = "noise_burst"
            elif persist:
                d = np.abs(np.nan_to_num(dev[main][max(0, onset - 15):min(n, max(end, onset + 60))]))
                pk = np.nanmedian(np.sort(d)[-max(3, len(d) // 8):])
                t20 = np.argmax(d >= .2 * pk) if pk > 0 else 0
                t80 = np.argmax(d >= .8 * pk) if pk > 0 else 0
                typ = "level_shift" if (t80 - t20) <= 12 else "drift"
            elif k_big <= 8:
                typ = "spike"
            else:
                typ = "noise_burst"
        # ---- which sensors, magnitude
        if typ in ("comm_error", "power_glitch"):
            vars_ = list(V)
        elif typ == "stuck":
            vars_ = stuck_v
        else:
            vars_ = [main]
        prim = vars_[0] if typ != "stuck" else ("T" if "T" in vars_ else vars_[0])
        sensor = ph.SENSORS[prim] if len(vars_) == 1 else ("Temperature / humidity probe" if set(vars_) == {"T", "RH"} else "Logger (all sensors)" if len(vars_) == 3 else " + ".join(ph.SENSORS[v] for v in vars_))
        dv = dev[prim]
        mag = float(dv[seg][np.nanargmax(np.abs(np.nan_to_num(dv[seg])))]) if np.isfinite(dv[seg]).any() else 0.0
        # true onset: stuck / stuck runs are only confirmed after the run length
        est_onset = onset
        if typ == "stuck":
            est_onset = max(0, onset - int(ev["F"][prim]["run"][onset, i] - 1)) if ev["F"][prim]["run"][onset, i] > 1 else onset
        p_early = np.nanmedian(ev["p_model"][i, onset:min(n, onset + 10)])
        conf = float(min(.99, max(p_early, .99 if (ev["hard"][i, seg].mean() > .5) else 0)))
        # ---- severity
        base = {"spike": "high", "level_shift": "high", "stuck": "high", "power_glitch": "high", "comm_error": "medium", "noise_burst": "medium", "drift": "medium"}[typ]
        lvl = SEV_ORDER.index(base)
        if typ in ("spike", "level_shift", "drift") and abs(mag) < 1.5 * ph.TOLERANCE[prim]:
            lvl -= 1
        if conf < .7:
            lvl -= 1
        sev = SEV_ORDER[max(0, lvl)]
        # ---- corrected values
        corr = []
        for v in vars_:
            k = end - 1 if not ongoing else n - 1
            k = min(max(k, 0), n - 1)
            obs = float(raw[v][i, k]) if np.isfinite(raw[v][i, k]) and raw[v][i, k] > -90 else None
            est = self._estimate(ev, i, v, k, onset)
            if est is not None:
                corr.append(dict(var=v, observed=obs, corrected=float(est)))
        # ---- checks & explanation
        chk = self._checks(panel, ev, i, seg, typ, vars_, prim, has_nb, nb_idx, nb_km, zpeak, mag, conf)
        text = self._sentence(ev, i, seg, typ, vars_, prim, has_nb, len(nb_idx), mag, onset, end)
        contrib = self._contrib(ev, i, seg, has_nb, zpeak)
        t = panel.time
        return dict(kind="fault", station=panel.stations.station.iloc[i], idx=i, city=panel.stations.city.iloc[i], type=typ, label=TYPE_LABEL[typ],
                    cause=CAUSE[typ], vars=vars_, var=prim, sensor=sensor, onset=int(est_onset), detect=int(detect), end=int(end), ongoing=bool(ongoing),
                    onset_time=str(t[est_onset]), detect_time=str(t[detect]), confidence=round(conf, 3), severity=sev,
                    magnitude=round(mag, 2), unit=ph.UNITS[prim], corrected=corr, explanation=text, checks=chk, contributions=contrib,
                    action=ACTION[typ].format(s=sensor.lower() if len(vars_) == 1 else "sensors"), n_neighbours=len(nb_idx),
                    neighbour_km=round(max(nb_km), 0) if nb_km else None, has_neighbours=has_nb)

    def _expected_series(self, ev, i, v):
        """What sensor v of station i should read at every minute, from its neighbours. None without neighbours."""
        m = self.m
        idx = m.nb[i][0]
        if not idx:
            return None
        med, _ = _nanmed_rows(ev["A"][v], idx)
        off = _interp24(m.off[v], ev["hour"]).T[i]
        cl = _interp24(m.clim_tab[v][i:i + 1], ev["hour"])[:, 0]
        if v == "RH":
            return np.minimum(100.0, cl * np.exp((med + off) / 100.0))
        return cl + med + off

    def _estimate(self, ev, i, v, k, onset):
        """Best estimate of what the sensor should read: neighbours, else own history + climatology."""
        m = self.m
        idx = m.nb[i][0]
        if idx:
            med, cnt = _nanmed_rows(ev["A"][v], idx)
            if np.isfinite(med[k]):
                off = np.interp(ev["hour"][k], np.arange(25.0), np.append(m.off[v][i], m.off[v][i, 0]))
                cl = np.interp(ev["hour"][k], np.arange(25.0), np.append(m.clim_tab[v][i], m.clim_tab[v][i, 0]))
                if v == "RH":
                    return float(min(100.0, cl * np.exp((med[k] + off) / 100.0)))
                return cl + med[k] + off
        cl = _interp24(m.clim_tab[v][i:i + 1], ev["hour"])[:, 0]
        j = max(0, onset - 15)
        last = ev["clean"][v][i, max(0, onset - 60):j + 1]
        last = last[np.isfinite(last)]
        if len(last) == 0:
            return None
        return float(last[-1] + cl[k] - cl[j])

    def _checks(self, panel, ev, i, seg, typ, vars_, prim, has_nb, nb_idx, nb_km, zpeak, mag, conf):
        c = []
        raw = ev["raw"]
        # 1 quick check
        msgs, bad = [], False
        if ev["comm"][i, seg].mean() > .2:
            bad = True; msgs.append(f"Missing or invalid packets for {int(ev['comm'][i, seg].sum())} of {seg.stop - seg.start} minutes.")
        for v in V:
            if ev["rng"][v][i, seg].any():
                bad = True; x = raw[v][i, seg][ev["rng"][v][i, seg]][0]; lo, hi = ph.RANGE[v]
                msgs.append(f"{ph.NAMES[v]} {_f(x, 1)} {ph.UNITS[v]} is outside the {lo:g} to {hi:g} limit.")
            elif ev["step"][v][i, seg].any():
                bad = True; k = seg.start + int(np.argmax(ev["step"][v][i, seg]))
                msgs.append(f"{ph.NAMES[v]} changed {_f(abs(raw[v][i, k] - raw[v][i, k - 1]), 1)} {ph.UNITS[v]} in 1 minute.")
            if ev["stuck"][v][i, seg].any():
                bad = True; msgs.append(f"{ph.NAMES[v]} identical for {int(ev['F'][v]['run'][seg, i].max()) + 1} readings in a row.")
        c.append(dict(id="qc", title="Quick check on the station", tag="range · jump · stuck · missing", state="fail" if bad else "pass",
                      text=" ".join(msgs[:2]) if bad else "No impossible values, sudden jumps, stuck or missing readings."))
        # 2 physics
        phm = float(np.nanmax(ev["ph"][i, seg])) if np.isfinite(ev["ph"][i, seg]).any() else 0.0
        td = ph.dewpoint(ev["clean"]["T"][i], ev["clean"]["RH"][i])
        if phm >= 1:
            k = seg.start + int(np.nanargmax(ev["ph"][i, seg]))
            c.append(dict(id="physics", title="Physics check", tag="T ↔ RH dew point", state="fail",
                          text=f"Temperature and humidity together imply a dew point of {_f(td[k], 1)} °C, moved {_sg(td[k] - np.nanmedian(td[max(0, k - 35):max(1, k - 5)]), 1)} °C in 30 minutes. Air moisture cannot change that fast."))
        else:
            c.append(dict(id="physics", title="Physics check", tag="T ↔ RH dew point", state="pass", text="Temperature, pressure and humidity are consistent. Dew point steady."))
        # 3 neighbours
        if has_nb:
            z = zpeak[prim]
            if z >= 5:
                d = self._dev_native(ev, i, prim)
                if ev["rng"][prim][i].any():               # masked out-of-range values: measure against the neighbour estimate
                    ex = self._expected_series(ev, i, prim)
                    if ex is not None:
                        d = np.where(ev["rng"][prim][i], raw[prim][i] - ex, d)
                k = seg.start + int(np.nanargmax(np.abs(np.nan_to_num(d[seg]))))
                c.append(dict(id="neighbour", title="Neighbour check", tag="buddy · 100 km", state="fail",
                              text=f"{ph.NAMES[prim]} is {_sg(d[k], ph.NAMES[prim] == 'Humidity' and 0 or 1)} {ph.UNITS[prim]} from {len(nb_idx)} stations within {max(nb_km):.0f} km. Only this station is off."))
            else:
                c.append(dict(id="neighbour", title="Neighbour check", tag="buddy · 100 km", state="warn",
                              text=f"Close to the {len(nb_idx)} stations within {max(nb_km):.0f} km. The fault shows in the station's own signals."))
        else:
            c.append(dict(id="neighbour", title="Neighbour check", tag="buddy · 100 km", state="na",
                          text="No other station within 100 km. Physics and AI only, so confidence is lower."))
        ts = float(np.nanmax(ev["ts"][i, seg])) if np.isfinite(ev["ts"][i, seg]).any() else 0
        c.append(dict(id="ai", title="AI check", tag="IsolationForest + autoencoder", state="fail" if ts >= 1 else "pass",
                      text=f"Unusualness score {ts:.2f} against this station's own learned pattern (1.0 = outside 99.9% of normal behaviour)."))
        return c

    def _sentence(self, ev, i, seg, typ, vars_, prim, has_nb, nn, mag, onset, end):
        N, U = ph.NAMES[prim].lower(), ph.UNITS[prim]
        who = f"{nn} nearby stations stayed normal" if has_nb else "no neighbour is available to confirm, so physics and AI decided"
        dur = end - onset
        if typ == "comm_error":
            return f"Flagged: packets are missing or invalid for {dur} minutes, so all three readings are unusable."
        if typ == "stuck":
            names = " and ".join(ph.NAMES[v].lower() for v in vars_)
            return f"Flagged: {names} have not changed for {dur} minutes" + (f" while {nn} nearby stations kept moving." if has_nb else ".")
        if typ == "noise_burst":
            return f"Flagged: {N} readings became erratic for {dur} minutes with no matching weather change, while {who}."
        if typ == "power_glitch":
            return f"Flagged: temperature, pressure and humidity jumped together at random for {dur} minutes, the signature of a power fault, while {who}."
        if typ == "spike":
            return f"Flagged: {N} jumped {_sg(mag, 1)} {U} for {max(dur, 1)} minutes and returned, while {who}."
        if typ == "drift":
            return f"Flagged: {N} has crept {_sg(mag, 1)} {U} away from the expected value over {dur} minutes, while {who}."
        return f"Flagged: {N} jumped {_sg(mag, 1)} {U} within minutes and stayed there, while {who}."

    def _contrib(self, ev, i, seg, has_nb, zpeak):
        a = min(1.0, max(zpeak.values()) / 12.0) if has_nb else 0.05
        b = min(1.0, float(np.nanmax(ev["ph"][i, seg])))
        c = 1.0 if (ev["hard"][i, seg].any() or ev["step_any"][i, seg].any()) else .1
        d = min(1.0, float(np.nanmax(ev["ts"][i, seg])) / 3.0) * .6
        raw = [("Difference from neighbours", a), ("Dew-point jump (physics)", b), ("Broke a station limit", c), ("Unusual for this station (AI)", d)]
        s = sum(x for _, x in raw) or 1
        return [dict(name=k, share=round(100 * x / s)) for k, x in raw]

    # ----------------------------------------------------------------- genuine events
    def _extreme(self, panel, ev, i, fault_mask):
        """Slow, large departures from the station's own climate that neighbours share: heatwave / cold spell."""
        a = np.nan_to_num(ev["A"]["T"][i])
        n = len(a)
        res = []
        nb_idx, nb_km = self.m.nb[i]
        if not nb_idx:
            return res
        act = (np.abs(a) >= 4.0) & ~fault_mask
        for (s, l) in _runs(act, 10):
            if l - s < 30:
                continue
            frac = float(np.mean([np.nanmedian(np.abs(ev["A"]["T"][j, s:l + 1])) >= 3 for j in nb_idx]))
            kind = "Heatwave" if a[s:l + 1].mean() > 0 else "Cold spell"
            nn = len(nb_idx)
            res.append(dict(kind="event", station=panel.stations.station.iloc[i], idx=i, city=panel.stations.city.iloc[i], type="event", label=kind, event_kind=kind,
                            onset=int(s), detect=int(s + 30), end=int(l + 1), ongoing=bool(l >= n - 2), onset_time=str(panel.time[s]), detect_time=str(panel.time[min(n - 1, s + 30)]),
                            confidence=round(min(.98, .72 + .25 * frac), 3), severity="ok", neighbours_agree=frac, n_neighbours=nn, neighbour_km=round(max(nb_km), 0),
                            explanation=f"Not a fault: temperature is {_sg(float(a[s:l + 1].mean()), 1)} \u00b0C from normal, and {int(round(frac * nn))} of {nn} nearby stations show the same departure. Real weather is kept, not deleted."))
        return res

    def _events(self, panel, ev, i, fault_mask):
        ts = ev["ts"][i]
        act = (ts >= 1.0) & ~fault_mask
        n = len(ts)
        cs = np.concatenate([[0], np.cumsum(act)])
        sm = np.array([cs[t + 1] - cs[max(0, t - 4)] for t in range(n)])
        res, t = [], 0
        nb_idx, nb_km = self.m.nb[i]
        while t < n:
            if sm[t] >= 3 and not fault_mask[t]:
                s = t; last = t
                while t < n and t - last < 15:
                    t += 1
                    if t < n and act[t]: last = t
                e = last + 1
                if fault_mask[s:e].any(): e = s + int(np.argmax(fault_mask[s:e])) or e
                res.append(self._event(panel, ev, i, s, e, nb_idx, nb_km))
                t = last + 15
            else:
                t += 1
        return res

    def _event(self, panel, ev, i, s, e, nb_idx, nb_km):
        seg = slice(s, e)
        cl = ev["clean"]
        def chg(v, w=60):
            a = np.nanmedian(cl[v][i, max(0, s - w):max(1, s - 5)]); b = np.nanmedian(cl[v][i, s:min(cl[v].shape[1], s + 40)])
            return b - a
        dT, dP, dRH = chg("T"), chg("P"), chg("RH")
        rh_now = np.nanmedian(cl["RH"][i, s:min(cl["RH"].shape[1], s + 60)])
        Tnow = np.nanmedian(cl["T"][i, s:min(cl["T"].shape[1], s + 60)])
        if dT >= 3:
            kind = "Heatwave"
        elif dT <= -2.5 and dRH >= 5 and abs(dP) < 4:
            kind = "Thunderstorm outflow" if s and (np.nanmax(np.abs(np.diff(cl['T'][i, s:min(cl['T'].shape[1], s + 15)]))) if e > s + 1 else 0) > .3 else "Cold front"
        elif dT <= -2.5:
            kind = "Cold front"
        elif rh_now >= 95 and dT <= 0:
            kind = "Fog"
        else:
            kind = "Weather change"
        frac = float(np.mean([np.nanmax(ev["ts"][j, seg]) >= .7 for j in nb_idx])) if nb_idx else None
        conf = min(.98, .72 + .25 * (frac if frac is not None else .3))
        nn = len(nb_idx)
        if nn:
            txt = f"Not a fault: temperature, pressure and humidity changed together, and {int(round(frac * nn))} of {nn} nearby stations saw the same change. Real weather is kept, not deleted."
        else:
            txt = "Unusual but physically consistent. No neighbour is available to confirm, so it is kept with a lower confidence."
        return dict(kind="event", station=panel.stations.station.iloc[i], idx=i, city=panel.stations.city.iloc[i], type="event", label=kind,
                    event_kind=kind, onset=int(s), detect=int(s), end=int(e), ongoing=False, onset_time=str(panel.time[s]), detect_time=str(panel.time[s]),
                    confidence=round(conf, 3), severity="ok", explanation=txt, neighbours_agree=frac, n_neighbours=nn,
                    neighbour_km=round(max(nb_km), 0) if nb_km else None)
