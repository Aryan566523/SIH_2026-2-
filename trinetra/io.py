"""Load your own AWS data (CSV) and run the detector on it."""
import numpy as np
import pandas as pd
from .engine import Panel

COLS = {"station": ["station", "station_id", "id"], "time": ["time", "timestamp", "datetime", "date_time"],
        "T": ["temperature", "temp", "t", "ta"], "P": ["pressure", "pres", "p", "slp"], "RH": ["humidity", "rh", "relative_humidity"]}


def _pick(df, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    raise KeyError(f"none of {names} found in columns {list(df.columns)}")


def load_csv(obs_csv, stations_csv):
    """obs_csv: one row per station per minute (station, time, temperature, pressure, humidity).
    stations_csv: station, lat, lon (elev optional). Missing minutes become NaN and are treated as communication gaps."""
    o = pd.read_csv(obs_csv)
    s = pd.read_csv(stations_csv)
    c = {k: _pick(o, v) for k, v in COLS.items()}
    o = o.rename(columns={c["station"]: "station", c["time"]: "time", c["T"]: "T", c["P"]: "P", c["RH"]: "RH"})
    o["time"] = pd.to_datetime(o["time"]).dt.floor("min")
    sc = {k: _pick(s, [k]) for k in ("station", "lat", "lon")}
    s = s.rename(columns={sc["station"]: "station", sc["lat"]: "lat", sc["lon"]: "lon"})
    if "elev" not in s.columns:
        s["elev"] = 0.0
    if "city" not in s.columns:
        s["city"] = s["station"]
    s = s.reset_index(drop=True)
    t = pd.date_range(o.time.min(), o.time.max(), freq="min")
    meas = {v: np.full((len(s), len(t)), np.nan) for v in ("T", "P", "RH")}
    for i, sid in enumerate(s.station):
        d = o[o.station == sid].drop_duplicates("time").set_index("time").reindex(t)
        for v in meas:
            meas[v][i] = d[v].values
    return Panel(s, t, meas)
