import argparse

from .config import load_config
from .optimize import optimize


def main():
    parser = argparse.ArgumentParser(description="Nonlinear compliant inverter topology optimization")
    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument("--output", help="Empty directory for results")
    parser.add_argument("--max-iter", type=int, help="Override the optimization iteration limit")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output:
        config["output"]["directory"] = args.output
    if args.max_iter is not None:
        config["optimization"]["max_iter"] = args.max_iter
    optimize(config)


if __name__ == "__main__":
    main()
