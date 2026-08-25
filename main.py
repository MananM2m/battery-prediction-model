import argparse
import subprocess
import sys


def run_generate(args):
	subprocess.check_call([sys.executable, "-m", "bt.generate_data", "--out", args.out])


def run_train(args):
	subprocess.check_call([sys.executable, "-m", "bt.train_and_save", "--data", args.data, "--out", args.out])


def run_serve(args):
	# run uvicorn programmatically
	import uvicorn
	uvicorn.run("bt.inference_app:app", host="0.0.0.0", port=args.port, reload=False)


def main():
	p = argparse.ArgumentParser(description="ML telemetry demo: generate/train/serve")
	sub = p.add_subparsers(dest="cmd")

	g = sub.add_parser("generate")
	g.add_argument("--out", default="data/telemetry.csv")

	t = sub.add_parser("train")
	t.add_argument("--data", default="data/telemetry.csv")
	t.add_argument("--out", default="models")

	s = sub.add_parser("serve")
	s.add_argument("--port", type=int, default=8000)

	args = p.parse_args()
	if args.cmd == "generate":
		run_generate(args)
	elif args.cmd == "train":
		run_train(args)
	elif args.cmd == "serve":
		run_serve(args)
	else:
		p.print_help()


if __name__ == "__main__":
	main()

