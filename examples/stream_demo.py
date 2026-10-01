"""Real-time use: push one minute of observations at a time.

    python examples/stream_demo.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
import numpy as np
from trinetra.evaluate import train_model
from trinetra.simulate import simulate, inject
from trinetra.stream import StreamMonitor

model, stations = train_model()
sim = simulate(stations, 330, seed=5, start="2026-06-10 06:00", events=False)
i = int(stations.index[stations.station == "AWS-PUN-003"][0])
inject(sim, i, "stuck", start=250, dur=70, var="P", r=np.random.default_rng(1))

WARM = 200                                              # first minutes only fill the monitor's history window
monitor = StreamMonitor(model, stations)
for k in range(len(sim.time)):
    packet = {s: (sim.meas["T"][j, k], sim.meas["P"][j, k], sim.meas["RH"][j, k]) for j, s in enumerate(stations.station)}
    for alert in monitor.push(sim.time[k], packet):     # a packet of None (or NaN values) means the station sent nothing
        if k >= WARM: print(sim.time[k], "UPDATED" if alert.get("updated") else "NEW", alert["station"], alert["label"], f"{alert['confidence']:.0%}", "-", alert["explanation"])
print(f"median processing time per minute for {len(stations)} stations: {np.median(monitor.latency_ms):.0f} ms")
