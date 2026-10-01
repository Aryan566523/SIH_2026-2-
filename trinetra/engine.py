"""TRINETRA detection engine.

Layers (all causal, so the same code runs on a batch or on a live stream):
  1. Quick QC      range / step / stuck / missing-or-sentinel (Zahumensky 2004, WMO No. 8)
  2. Physics       dew-point coherence between T and RH
  3. Temporal AI   IsolationForest + neural autoencoder on multivariate residual features
  4. Spatial       buddy check against neighbours inside 100 km (removes real weather)
  5. Fusion        gradient-boosted classifier learns how to weigh 1-4 -> P(sensor/data fault)
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from . import physics as ph

V = ph.VARS
FLOOR = {"T": .35, "P": .25, "RH": 2.0}       # min sigma for spatial z-scores
STUCK_MIN = {"T": 15, "P": 15, "RH": 20}       # identical readings in a row
P_THR = 0.7                                    # fault probability threshold
NB_RADIUS, NB_MAX = 100.0, 6


@dataclass
class Panel:
    """A block of 1-minute observations for many stations."""
    stations: pd.DataFrame
    time: pd.DatetimeIndex
    meas: dict            # var -> (n_stations, n) raw values (NaN / sentinels allowed)

    @classmethod
    def from_sim(cls, sim):
        return cls(sim.stations, sim.time, sim.meas)

    def tail(self, k):
        return Panel(self.stations, self.time[-k:], {v: a[:, -k:] for v, a in self.meas.items()})


def _hour(time):
    return time.hour.values + time.minute.values / 60.0


def _interp24(tab, hour):
    """tab (ns,24) hour-of-day table -> (n, ns) interpolated (circular)."""
    x = np.arange(25.0)
    out = np.empty((len(hour), tab.shape[0]))
    for i in range(tab.shape[0]):
        out[:, i] = np.interp(hour, x, np.append(tab[i], tab[i, 0]))
    return out


def neighbours(stations):
    lon, lat = stations.lon.values, stations.lat.values
    out = []
    for i in range(len(stations)):
        d = ph.haversine(lon[i], lat[i], lon, lat)
        d[i] = np.inf
        idx = np.argsort(d)[:NB_MAX]
        idx = [int(j) for j in idx if d[j] <= NB_RADIUS]
        out.append((idx, [float(d[j]) for j in idx]))
    return out


def _nanmed_rows(arr, idx):
    """median over rows `idx` of arr (ns,n) -> (n,), plus count of valid values."""
    if len(idx) == 0:
        return np.full(arr.shape[1], np.nan), np.zeros(arr.shape[1], int)
    sub = arr[idx]
    cnt = np.isfinite(sub).sum(0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.nanmedian(sub, axis=0)
    m[cnt < 2] = np.nan
    return m, cnt


def _nb_spread(arr, idx):
    """Robust disagreement (1.4826*MAD) between neighbours at each minute."""
    if len(idx) < 3:
        return np.zeros(arr.shape[1])
    import warnings
    sub = arr[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.nanmedian(sub, axis=0)
        mad = np.nanmedian(np.abs(sub - m), axis=0)
    return np.nan_to_num(1.4826 * mad, nan=0.0)


class Trinetra:
    def __init__(self, seed=0, if_trees=120):
        self.seed = seed
        self.if_trees = if_trees
        self.fitted = False

    # ---------------------------------------------------------------- feature computation
    def _prep(self, panel):
        raw = {v: np.asarray(panel.meas[v], float) for v in V}
        nanm = {v: ~np.isfinite(raw[v]) for v in V}
        sentinel = (np.nan_to_num(raw["T"], nan=0) <= -90) | (np.nan_to_num(raw["P"], nan=5) <= 1) | (np.nan_to_num(raw["RH"], nan=0) <= -90)
        comm = nanm["T"] | nanm["P"] | nanm["RH"] | sentinel
        clean, rng = {}, {}
        for v in V:
            lo, hi = ph.RANGE[v]
            bad_rng = (raw[v] < lo) | (raw[v] > hi)
            rng[v] = bad_rng & ~comm
            clean[v] = np.where(nanm[v] | sentinel | bad_rng, np.nan, raw[v])
        return raw, clean, comm, rng

    def _clim(self, hour):
        return {v: _interp24(self.clim_tab[v], hour) for v in V}

    def _features(self, clean, hour):
        """Causal per-station feature blocks. Every array is (n, ns)."""
        CL = self._clim(hour)
        F, aux = {}, {}
        X = {v: pd.DataFrame(clean[v].T) for v in V}
        for v in V:
            x, cl = X[v], pd.DataFrame(CL[v])
            d1 = x.diff(1)
            d5 = x - x.shift(5) - (cl - cl.shift(5))
            base = x.shift(1).rolling(30, min_periods=15).median()
            innov = x - base - (cl - cl.shift(15))
            dsd = d1.rolling(10, min_periods=6).std() / ph.NOISE[v]
            eq = (x.values == x.shift(1).values) & np.isfinite(x.values)
            run = np.zeros(x.shape)
            for t in range(1, x.shape[0]):
                run[t] = (run[t - 1] + 1) * eq[t]
            F[v] = dict(d1=d1.values, d5=d5.values, innov=innov.values, dsd=dsd.values, run=run)
        Td = pd.DataFrame(dewpoint_arr(clean))
        tdi = Td - Td.shift(1).rolling(30, min_periods=15).median()
        F["Td"] = dict(tdi=tdi.values, d1=Td.diff(1).values)
        return F

    def _matrix(self, F):
        cols = []
        for v in V:
            f = F[v]
            cols += [f["d1"], f["d5"], f["innov"], f["dsd"], np.log1p(f["run"])]
        cols += [F["Td"]["tdi"], F["Td"]["d1"]]
        n, ns = cols[0].shape
        return np.stack(cols, axis=-1).reshape(n * ns, len(cols)), (n, ns)

    def _scale(self, M):
        return np.clip((M - self.med) / self.iqr, -40, 40)

    # ---------------------------------------------------------------- training
    def fit(self, panel, verbose=False):
        """Learn normal behaviour from clean historical data (genuine events allowed)."""
        self.stations = panel.stations
        self.nb = neighbours(panel.stations)
        hour = _hour(panel.time)
        raw, clean, comm, rng = self._prep(panel)
        # climatology per station and hour
        self.clim_tab = {}
        for v in V:
            tab = np.zeros((len(self.stations), 24))
            hh = np.floor(hour).astype(int)
            for h in range(24):
                with np.errstate(all="ignore"):
                    tab[:, h] = np.nanmean(clean[v][:, hh == h], axis=1)
            self.clim_tab[v] = tab
        F = self._features(clean, hour)
        self.dsd_ref = {v: float(np.nanmedian(F[v]["dsd"])) for v in V}
        M, shape = self._matrix(F)
        ok = np.isfinite(M).all(1)
        Mok = M[ok]
        self.med = np.median(Mok, 0)
        q = np.percentile(Mok, [25, 75], axis=0)
        self.iqr = np.maximum(q[1] - q[0], np.percentile(np.abs(Mok - self.med), 90, axis=0) * .5 + 1e-3)
        S = self._scale(M)
        rs = np.random.default_rng(self.seed)
        pick = rs.choice(np.where(ok)[0], size=min(70000, ok.sum()), replace=False)
        self.iforest = IsolationForest(n_estimators=self.if_trees, max_samples=512, random_state=self.seed, n_jobs=1).fit(S[pick])
        self.ae = MLPRegressor(hidden_layer_sizes=(14, 5, 14), activation="tanh", max_iter=80, random_state=self.seed,
                               early_stopping=True, validation_fraction=.1, n_iter_no_change=6)
        self.ae.fit(np.clip(S[pick], -10, 10), np.clip(S[pick], -10, 10))
        s_if = -self.iforest.score_samples(S[ok])
        s_ae = self._ae_err(S[ok])
        self.if_ref = float(np.percentile(s_if, 99.9)); self.if_base = float(np.median(s_if))
        self.ae_ref = float(np.percentile(s_ae, 99.9))
        # spatial reference: hour-of-day offsets and robust sigma of station-minus-neighbour residuals
        self.off, self.sig, self.sigI = {}, {}, {}
        res, resI, _ = self._spatial_raw(clean, F, hour)
        hh = np.floor(hour).astype(int)
        for v in V:
            ns = len(self.stations)
            off = np.zeros((ns, 24)); sg = np.full(ns, FLOOR[v]); sgI = np.full(ns, FLOOR[v] * .5)
            for i in range(ns):
                r = res[v][i]
                if np.isfinite(r).sum() < 500:
                    continue
                for h in range(24):
                    z = r[hh == h]; z = z[np.isfinite(z)]
                    off[i, h] = np.median(z) if len(z) else 0
                r2 = r - np.interp(hour, np.arange(25.0), np.append(off[i], off[i, 0]))
                a2 = np.abs(r2[np.isfinite(r2)] - np.nanmedian(r2))
                sg[i] = max(FLOOR[v], 1.4826 * np.median(a2), np.percentile(a2, 99) / 2.576)
                ri = resI[v][i]
                sgI[i] = max(FLOOR[v] * .5, 1.4826 * np.nanmedian(np.abs(ri - np.nanmedian(ri))))
            self.off[v], self.sig[v], self.sigI[v] = off, sg, sgI
        self.fitted = True
        return self

    def _ae_err(self, S):
        Sc = np.clip(S, -10, 10)
        return np.mean((self.ae.predict(Sc) - Sc) ** 2, axis=1)

    # ---------------------------------------------------------------- spatial residuals
    def _anom(self, clean, hour):
        """Value minus the station's own hour-of-day climatology, so static offsets (elevation, siting) cancel."""
        CL = self._clim(hour)
        out = {v: clean[v] - CL[v].T for v in V}
        # humidity is compared as a ratio: RH_i/RH_j = es(Td_i)/es(Td_j) is constant, whatever the temperature
        out["RH"] = 100.0 * (np.log(np.maximum(clean["RH"], 0.5)) - np.log(np.maximum(CL["RH"].T, 0.5)))
        return out

    def _spatial_raw(self, clean, F, hour, exclude=None):
        res, resI, spr = {}, {}, {}
        A = self._anom(clean, hour)
        for v in V:
            ns, n = clean[v].shape
            r = np.full((ns, n), np.nan); ri = np.full((ns, n), np.nan); sp = np.zeros((ns, n))
            innov = F[v]["innov"].T
            Ar = A[v] if exclude is None else np.where(exclude, np.nan, A[v])
            innov_r = innov if exclude is None else np.where(exclude, np.nan, innov)
            for i, (idx, _) in enumerate(self.nb):
                m, _ = _nanmed_rows(Ar, idx)
                sp[i] = _nb_spread(Ar, idx)
                r[i] = A[v][i] - m
                mi, _ = _nanmed_rows(innov_r, idx)
                ri[i] = innov[i] - mi
            res[v], resI[v], spr[v] = r, ri, sp
        return res, resI, spr

    # ---------------------------------------------------------------- scoring
    def features(self, panel):
        """Compute all per-minute evidence for a panel. Returns dict of (ns, n) arrays."""
        assert self.fitted
        hour = _hour(panel.time)
        raw, clean, comm, rng = self._prep(panel)
        F = self._features(clean, hour)
        M, (n, ns) = self._matrix(F)
        ok = np.isfinite(M).all(1)
        S = self._scale(M)
        ts_if = np.zeros(len(M)); ts_ae = np.zeros(len(M))
        if ok.any():
            ts_if[ok] = np.maximum(0, (-self.iforest.score_samples(S[ok]) - self.if_base) / (self.if_ref - self.if_base))
            ts_ae[ok] = self._ae_err(S[ok]) / self.ae_ref
        ts_if = ts_if.reshape(n, ns).T; ts_ae = ts_ae.reshape(n, ns).T
        ts = np.maximum(ts_if, ts_ae)
        out = dict(raw=raw, clean=clean, comm=comm, rng=rng, ts_if=ts_if, ts_ae=ts_ae, ts=ts, hour=hour)
        # quick-check flags
        step = {v: np.nan_to_num(np.abs(F[v]["d1"]).T, nan=0) > ph.STEP[v] for v in V}
        stuck = {v: F[v]["run"].T >= STUCK_MIN[v] for v in V}
        stuck["RH"] &= np.nan_to_num(clean["RH"], nan=100) < 98
        out.update(step=step, stuck=stuck, F=F)
        out["stuck_any"] = stuck["T"] | stuck["P"] | stuck["RH"]
        out["step_any"] = step["T"] | step["P"] | step["RH"]
        out["ph"] = np.nan_to_num(np.abs(F["Td"]["tdi"]).T, nan=0) / 6.0
        out["dsd_max"] = np.nan_to_num(np.max([F[v]["dsd"].T for v in V], axis=0), nan=0)
        out.update(self._spatial_evidence(clean, F, hour, ts))
        return out

    def _spatial_evidence(self, clean, F, hour, ts, exclude=None):
        """Buddy-check evidence. `exclude` (ns, n) marks neighbours to ignore (e.g. already flagged faulty)."""
        res, resI, spr = self._spatial_raw(clean, F, hour, exclude)
        zL, zI = {}, {}
        for v in V:
            offt = _interp24(self.off[v], hour).T
            # when neighbours disagree with each other the reference is unreliable, so widen the yardstick
            zL[v] = (res[v] - offt) / np.sqrt(self.sig[v][:, None] ** 2 + spr[v] ** 2)
            zI[v] = resI[v] / self.sigI[v][:, None]
        cnt = np.zeros(clean["T"].shape, int)
        A = self._anom(clean, hour)
        for i, (idx, _) in enumerate(self.nb):
            a = A["T"].copy() if exclude is None else np.where(exclude, np.nan, A["T"])
            _, c = _nanmed_rows(a, idx); cnt[i] = c
        nbt = np.zeros_like(ts)
        for i, (idx, _) in enumerate(self.nb):
            if idx:
                w = ts if exclude is None else np.where(exclude, np.nan, ts)
                with np.errstate(all="ignore"):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        nbt[i] = np.nan_to_num(np.nanmedian(w[idx], axis=0))
        return dict(zL=zL, zI=zI, nn=cnt, has_nb=(cnt >= 2), A=A, nb_ts=nbt)

    def fusion_matrix(self, ev, isolate=False):
        ns, n = ev["ts"].shape
        hn = ev["has_nb"] & (not isolate)
        z = lambda a: np.where(hn, np.minimum(np.nan_to_num(np.abs(a), nan=0), 40), 0)
        cols = [np.minimum(ev["ts_if"], 8), np.minimum(ev["ts_ae"], 8), (ev["comm"] | ev["rng"]["T"] | ev["rng"]["P"] | ev["rng"]["RH"]).astype(float),
                ev["stuck_any"].astype(float), ev["step_any"].astype(float), np.minimum(ev["ph"], 6),
                np.minimum(ev["dsd_max"], 30)]
        cols += [z(ev["zL"][v]) for v in V] + [z(ev["zI"][v]) for v in V]
        cols += [hn.astype(float), np.where(hn, np.minimum(ev["nb_ts"], 8), 0)]
        return np.stack([c.reshape(-1) for c in cols], axis=1)

    def calibrate(self, panel, anom, warmup=300, verbose=False):
        """Fit the learned fusion on a *validation* panel with injected faults."""
        ev = self.features(panel)
        ns, n = ev["ts"].shape
        y = (np.asarray(anom) != "").reshape(-1)
        w = np.ones(ns * n)
        keep = np.zeros((ns, n), bool); keep[:, warmup:] = True
        w[~keep.reshape(-1)] = 0
        # ambiguous rows (slow drifts before they become visible) get no weight
        for i in range(ns):
            a = np.asarray(anom[i]); t = 0
            while t < n:
                if a[t] == "":
                    t += 1; continue
                j = t
                while j < n and a[j] == a[t]: j += 1
                if a[t] == "drift": w[i * n + t: i * n + t + (j - t) // 3] = 0
                w[i * n + j: i * n + min(n, j + 8)] = 0
                t = j
        self.fusion = self._fit_fusion(ev, y, w)
        # stage 2: recompute the buddy check with neighbours that stage 1 already flagged left out
        ev2 = self._second_pass(panel, ev, self.fusion)
        self.fusion2 = self._fit_fusion(ev2, y, w)
        return self

    def _fit_fusion(self, ev, y, w):
        X = self.fusion_matrix(ev)
        X2 = self.fusion_matrix(ev, isolate=True)
        Xs = np.vstack([X, X2]); ys = np.concatenate([y, y]); ws = np.concatenate([w, w * .35])
        m = ws > 0
        cw = np.where(ys, 3.0, 1.0)
        clf = HistGradientBoostingClassifier(max_depth=5, learning_rate=.08, max_iter=180, random_state=self.seed,
                                             l2_regularization=1.0, min_samples_leaf=40)
        clf.fit(Xs[m], ys[m], sample_weight=(ws * cw)[m])
        return clf

    def _second_pass(self, panel, ev, clf):
        ns, n = ev["ts"].shape
        p = clf.predict_proba(self.fusion_matrix(ev))[:, 1].reshape(ns, n)
        hard = ev["comm"] | ev["rng"]["T"] | ev["rng"]["P"] | ev["rng"]["RH"] | ev["stuck_any"]
        flagged = (np.where(hard, .99, p) >= .5).astype(float)
        # a neighbour stays excluded for 30 minutes after it was last flagged
        ex = pd.DataFrame(flagged.T).rolling(30, min_periods=1).max().values.T > 0
        ev2 = dict(ev)
        ev2.update(self._spatial_evidence(ev["clean"], ev["F"], ev["hour"], ev["ts"], exclude=ex))
        ev2["excluded"] = ex
        return ev2

    def score(self, panel):
        """Full evidence + fault probability p (ns, n)."""
        ev = self.features(panel)
        ev = self._second_pass(panel, ev, self.fusion)
        ns, n = ev["ts"].shape
        p = self.fusion2.predict_proba(self.fusion_matrix(ev))[:, 1].reshape(ns, n)
        hard = ev["comm"] | ev["rng"]["T"] | ev["rng"]["P"] | ev["rng"]["RH"] | ev["stuck_any"]
        ev["p_model"] = p
        ev["hard"] = hard
        # near-certain errors raise an alert at once: a value outside the physical range, or a jump beyond the step limit
        ev["instant"] = ev["rng"]["T"] | ev["rng"]["P"] | ev["rng"]["RH"] | (ev["step_any"] & (p >= .5))
        ev["p"] = np.where(hard, np.maximum(p, .99), p)
        return ev


def dewpoint_arr(clean):
    """(ns,n) -> (n,ns) dew point from cleaned T and RH."""
    return ph.dewpoint(clean["T"].T, clean["RH"].T)
