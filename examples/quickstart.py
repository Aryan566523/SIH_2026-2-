"""Quick start: train, inject faults into a simulated network, print alerts.

    python examples/quickstart.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from trinetra.evaluate import train_model
from trinetra.simulate import simulate, inject
from trinetra.engine import Panel
from trinetra.incidents import Analyzer

model, stations = train_model()                         # ~40 s: 8 days clean history + 3 days for the fusion stage
sim = simulate(stations, 600, seed=3, start="2026-06-10 06:00", events=False)

# The example from the problem statement: one station suddenly reports ~55 C while neighbours stay normal.
delhi = int(stations.index[stations.station == "AWS-DEL-014"][0])
inject(sim, delhi, "level_shift", start=520, dur=60, var="T", sign=1, mag=17.0)

panel = Panel.from_sim(sim)
evidence = model.score(panel)                           # per-minute fault probability + all evidence
alerts = Analyzer(model).analyse(panel, evidence)       # incidents with cause, confidence, explanation, fix

for a in alerts:
    if a["kind"] == "fault":
        print(f"{a['station']}  {a['label']}  confidence {a['confidence']:.0%}  severity {a['severity']}")
        print("  root cause :", a["cause"])
        print("  why        :", a["explanation"])
        for c in a["checks"]:
            print(f"  [{c['state']:>4}] {c['title']}: {c['text']}")
        for c in a["corrected"]:
            print(f"  corrected  : {c['var']} observed {c['observed']} -> {c['corrected']:.1f}")
        print("  action     :", a["action"])
