"""Real-time use: feed one minute of observations at a time and get alerts back."""
import time
import numpy as np
import pandas as pd
from .engine import Panel, P_THR
from .incidents import Analyzer, detect_runs


class StreamMonitor:
    """Keeps a rolling window per station and runs the same pipeline on every new minute.
    `push` returns alerts that were raised by that minute."""

    def __init__(self, model, stations, window=150):
        self.m, self.st, self.W = model, stations, window
        self.buf = {v: np.full((len(stations), 0), np.nan) for v in ("T", "P", "RH")}
        self.times = []
        self.index = {s: k for k, s in enumerate(stations.station)}
        self.an = Analyzer(model)
        self.latency_ms = []
        self.seen = {}

    def push(self, ts, obs):
        """obs: {station_id: (T, P, RH)}; a missing station or None means no packet arrived."""
        col = {v: np.full((len(self.st), 1), np.nan) for v in self.buf}
        for sid, vals in obs.items():
            if vals is None:
                continue
            k = self.index[sid]
            for v, x in zip(("T", "P", "RH"), vals):
                col[v][k, 0] = np.nan if x is None else x
        for v in self.buf:
            self.buf[v] = np.concatenate([self.buf[v], col[v]], axis=1)[:, -self.W:]
        self.times = (self.times + [pd.Timestamp(ts)])[-self.W:]
        if len(self.times) < 90:
            return []
        t0 = time.perf_counter()
        P = Panel(self.st, pd.DatetimeIndex(self.times), self.buf)
        ev = self.m.score(P)
        n = len(self.times)
        alerts = []
        recent = np.where(((ev["p"][:, -6:] >= P_THR) | ev["instant"][:, -6:]).any(axis=1))[0]
        for i in recent:
            for (o, d, e, og) in detect_runs(ev["p"][i], instant=ev["instant"][i]):
                key = (int(i), str(self.times[d]))
                if d == n - 1 or (e >= n - 1 and n - 1 - d in (15, 30)):   # first raise, then re-check while it is still open
                    inc = self.an._fault(P, ev, int(i), o, d, e, og)
                    if d != n - 1:
                        if inc["type"] == self.seen.get(key):
                            continue
                        inc["updated"] = True                              # the diagnosis was refined (e.g. spike -> stuck)
                    self.seen[key] = inc["type"]
                    alerts.append(inc)
        self.latency_ms.append((time.perf_counter() - t0) * 1000)
        return alerts
