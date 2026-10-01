"""Command line:  python -m trinetra <command>

  train      train on simulated clean history and save model.pkl
  demo       score a simulated network with injected faults and print the alerts
  evaluate   full evaluation (accuracy vs baselines, stress tests) -> results/evaluation.json
  detect     run on your own CSV files
  export     regenerate the website data (data.js)
"""
import argparse, pickle, sys, os, json, warnings


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="trinetra", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("train"); a.add_argument("--out", default="model.pkl")
    a = sub.add_parser("demo"); a.add_argument("--model", default=None); a.add_argument("--days", type=int, default=2); a.add_argument("--seed", type=int, default=7)
    a = sub.add_parser("evaluate"); a.add_argument("--model", default=None); a.add_argument("--out", default="results")
    a = sub.add_parser("detect"); a.add_argument("obs"); a.add_argument("stations"); a.add_argument("--history", required=True, help="clean 1-minute history CSV (3+ days) of the same stations, used to learn each station's normal pattern"); a.add_argument("--model", default=None); a.add_argument("--out", default="alerts.json")
    a = sub.add_parser("export"); a.add_argument("--model", default=None); a.add_argument("--out", default=".")
    args = ap.parse_args(argv)

    from .evaluate import train_model, run_evaluation
    from .simulate import make_network, simulate, inject_random
    from .engine import Panel
    from .incidents import Analyzer

    def get_model():
        if getattr(args, "model", None) and os.path.exists(args.model):
            return pickle.load(open(args.model, "rb"))
        print("training on simulated clean history (about 40 s) ...", flush=True)
        return train_model(0)

    if args.cmd == "train":
        pickle.dump(train_model(0), open(args.out, "wb")); print("saved", args.out)
    elif args.cmd == "demo":
        m, st = get_model()
        sim = simulate(st, 1440 * args.days, seed=args.seed); inject_random(sim, seed=args.seed + 1, per_station_day=1.0)
        P = Panel.from_sim(sim); ev = m.score(P); inc = Analyzer(m).analyse(P, ev)
        for x in sorted(inc, key=lambda x: x["detect"])[:25]:
            if x["kind"] == "fault":
                print(f"[{x['detect_time'][5:16]}] {x['severity'].upper():6s} {x['station']}  {x['label']:<18s} conf {x['confidence']:.0%}  sensor: {x['sensor']}\n    {x['explanation']}\n    action: {x['action']}")
            else:
                print(f"[{x['detect_time'][5:16]}] EVENT  {x['station']}  {x['label']} (kept, real weather)")
        print(f"\n{sum(x['kind'] == 'fault' for x in inc)} fault alerts, {sum(x['kind'] == 'event' for x in inc)} genuine events; injected faults: {len(sim.anomalies)}")
    elif args.cmd == "evaluate":
        from .report import build_report
        m, st = get_model(); build_report(args.out, model=m, stations=st)
    elif args.cmd == "detect":
        from .io import load_csv
        from .engine import Trinetra
        base, _ = get_model()                     # the fusion stage is station-independent, so it is reused
        H = load_csv(args.history, args.stations)
        m = Trinetra().fit(H)
        m.fusion, m.fusion2 = base.fusion, base.fusion2
        P = load_csv(args.obs, args.stations)
        ev = m.score(P); inc = Analyzer(m).analyse(P, ev)
        json.dump(inc, open(args.out, "w"), indent=1, default=str)
        print(f"{sum(x['kind'] == 'fault' for x in inc)} fault alerts, {sum(x['kind'] == 'event' for x in inc)} genuine events -> {args.out}")
    elif args.cmd == "export":
        from .export_web import export
        m, st = get_model(); export(m, st, outdir=args.out)


if __name__ == "__main__":
    main()
