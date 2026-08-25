"""Command-line interface for the battery_prediction package."""
import argparse
import sys


def run_generate(args):
    from battery_prediction.data import generator

    generator.cli_main(args)


def run_train(args):
    from battery_prediction.models import train

    train.cli_main(args)


def run_evaluate(args):
    from battery_prediction.evaluate import evaluate

    evaluate.cli_main(args)


def run_serve(args):
    from battery_prediction.api import app as api_app
    import uvicorn

    uvicorn.run("battery_prediction.api.app:app", host="0.0.0.0", port=args.port, reload=False)


def main(argv=None):
    p = argparse.ArgumentParser(prog="battery_prediction", description="Battery telemetry ML toolkit")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("generate")
    g.add_argument("--out", default="data/telemetry.csv")
    g.add_argument("--n", type=int, default=2000)
    g.add_argument("--seed", type=int, default=None)

    t = sub.add_parser("train")
    t.add_argument("--data", default="data/telemetry.csv")
    t.add_argument("--out", default="models")
    t.add_argument("--seed", type=int, default=42)

    e = sub.add_parser("evaluate")
    e.add_argument("--data", default="data/telemetry.csv")
    e.add_argument("--models", default="models")

    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    if args.cmd == "generate":
        run_generate(args)
    elif args.cmd == "train":
        run_train(args)
    elif args.cmd == "evaluate":
        run_evaluate(args)
    elif args.cmd == "serve":
        run_serve(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
