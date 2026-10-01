"""Run on your own CSV files. This script first writes sample files so you can see the format.

    python examples/custom_csv.py

Format (1-minute data):
  observations.csv : station,time,temperature,pressure,humidity
  stations.csv     : station,lat,lon[,elev]
  history.csv      : same as observations.csv, 3+ clean days of the same stations
"""
import sys, os, warnings, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
from trinetra.simulate import make_network, simulate, inject_random

here = os.path.dirname(__file__)
st = make_network(0)
keep = st[st.city.isin(["DEL", "PUN"])].reset_index(drop=True)          # two clusters keep the sample small
hist = simulate(keep, 1440 * 4, seed=1); live = simulate(keep, 720, seed=2, start="2026-06-14 00:00"); inject_random(live, seed=3, per_station_day=3, warmup=200)
for name, sim in (("history", hist), ("observations", live)):
    df = sim.to_long().rename(columns={"T": "temperature", "P": "pressure", "RH": "humidity"}).drop(columns="anomaly")
    df.to_csv(os.path.join(here, f"{name}.csv"), index=False)
keep[["station", "lat", "lon", "elev"]].to_csv(os.path.join(here, "stations.csv"), index=False)
print("wrote history.csv, observations.csv, stations.csv in", here)
subprocess.run([sys.executable, "-m", "trinetra", "detect", os.path.join(here, "observations.csv"), os.path.join(here, "stations.csv"),
                "--history", os.path.join(here, "history.csv"), "--out", os.path.join(here, "alerts.json")], cwd=os.path.join(here, ".."))
