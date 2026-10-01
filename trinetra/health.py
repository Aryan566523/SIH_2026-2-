"""Sensor health and maintenance prediction from the spatial residual of each sensor."""
import numpy as np
from . import physics as ph
from .engine import V


def _hourly(x, mask=None):
    n = len(x) // 60 * 60
    x = x[-n:].reshape(-1, 60)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmedian(x, axis=1)


def sensor_health(model, ev, panel, incidents, analyzer, hours=72):
    """Return {station: {var: {...}, 'overall': ...}} using the last `hours` of data."""
    ns, n = ev["p"].shape
    W = min(n, hours * 60)
    out = {}
    for i in range(ns):
        st = panel.stations.station.iloc[i]
        rec = {}
        for v in V:
            fault_min = np.zeros(n, bool)
            for x in incidents:
                if x["kind"] == "fault" and x["idx"] == i and v in x["vars"]:
                    fault_min[x["onset"]:x["end"]] = True
            frac = float(fault_min[-min(n, 1440):].mean())
            dev = analyzer._dev_native(ev, i, v)
            bias = slope = None
            if dev is not None and np.isfinite(dev[-W:]).sum() > 120:
                h = _hourly(dev[-W:])                          # hourly medians resist spikes; a growing bias shows up as a slope
                ok = np.isfinite(h)
                if ok.sum() >= 6:
                    bias = float(np.nanmedian(h[-12:]))
                    bias = bias if np.isfinite(bias) else None
                    if ok.sum() >= 24 and bias is not None:
                        t = np.arange(len(h))[ok] / 24.0
                        slope = float(np.polyfit(t, h[ok], 1)[0])       # native units per day
            noise = float(np.nanmedian(ev["F"][v]["dsd"].T[i, -min(n, 1440):]) / max(model.dsd_ref[v], 1e-6))
            tol = ph.TOLERANCE[v]
            days = None
            if bias is not None and slope is not None and abs(slope) > 0.03 * tol and np.sign(slope) == np.sign(bias if abs(bias) > 0.15 * tol else slope):
                days = max(0.0, (tol - abs(bias)) / abs(slope))
            score = 100 - min(100, 60 * min(1, frac * 4) + 25 * min(1, abs(bias or 0) / tol) + 15 * min(1, max(0, noise - 1.5) / 3)
                              + (10 if (days is not None and days < 7) else 0))
            status = "ok" if score >= 80 else "watch" if score >= 55 else "degraded"
            rec[v] = dict(score=round(score), status=status, bias=None if bias is None else round(bias, 2), drift_per_day=None if slope is None else round(slope, 3),
                          days_to_limit=None if days is None else round(days, 1), fault_fraction_24h=round(frac, 3), noise_ratio=round(noise, 2), unit=ph.UNITS[v])
        worst = min(rec.values(), key=lambda r: r["score"])
        rec["overall"] = dict(score=worst["score"], status=worst["status"])
        out[st] = rec
    return out


def maintenance_note(rec):
    """Plain-language maintenance advice for one station's health record."""
    notes = []
    for v in V:
        r = rec[v]
        if r["days_to_limit"] is not None and r["days_to_limit"] < 30:
            notes.append(f"{ph.SENSORS[v]} is drifting {abs(r['drift_per_day']):.2f} {r['unit']} per day and is expected to exceed its {ph.TOLERANCE[v]:g} {r['unit']} tolerance in about {r['days_to_limit']:.1f} days. Schedule recalibration.")
        elif r["status"] != "ok":
            notes.append(f"{ph.SENSORS[v]} health {r['score']}/100. Inspect at the next visit.")
    return notes or ["No maintenance needed."]
