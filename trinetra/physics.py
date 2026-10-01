"""Atmospheric physics helpers and station limits used by every layer."""
import numpy as np

VARS = ("T", "P", "RH")
NAMES = {"T": "Temperature", "P": "Pressure", "RH": "Humidity"}
UNITS = {"T": "°C", "P": "hPa", "RH": "%"}
SENSORS = {"T": "Temperature probe", "P": "Barometer", "RH": "Humidity sensor"}

# Hard limits (WMO No. 8 / Zahumensky 2004 style quality-control rules), tuned for India.
RANGE = {"T": (-15.0, 50.0), "P": (500.0, 1080.0), "RH": (0.0, 100.0)}
# Largest believable 1-minute change.
STEP = {"T": 3.0, "P": 1.5, "RH": 20.0}
# Sensor accuracy tolerance used for maintenance prediction.
TOLERANCE = {"T": 1.0, "P": 1.0, "RH": 5.0}
# Typical sensor noise (1 sigma) used as a floor for normalisation.
NOISE = {"T": 0.08, "P": 0.05, "RH": 0.5}


def sat_vp(T):
    """Saturation vapour pressure in hPa (Magnus / Bolton 1980)."""
    T = np.asarray(T, dtype=float)
    return 6.112 * np.exp(17.67 * T / (T + 243.5))


def vapour_pressure(T, RH):
    return np.asarray(RH, dtype=float) / 100.0 * sat_vp(T)


def dewpoint(T, RH):
    e = np.maximum(vapour_pressure(T, np.maximum(RH, 0.5)), 1e-6)
    g = np.log(e / 6.112)
    return 243.5 * g / (17.67 - g)


def rh_from_dew(T, Td):
    return np.clip(100.0 * sat_vp(Td) / sat_vp(T), 0.0, 100.0)


def station_pressure(elev_m):
    """Standard-atmosphere station pressure (hPa) for an elevation."""
    return 1013.25 * (1 - 2.25577e-5 * elev_m) ** 5.25588


def haversine(lon1, lat1, lon2, lat2):
    r = np.pi / 180.0
    dl, dn = (lat2 - lat1) * r, (lon2 - lon1) * r
    a = np.sin(dl / 2) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin(dn / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))
