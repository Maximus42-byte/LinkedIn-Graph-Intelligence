"""Command line entry point for the offline-first pipeline."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile graph analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--source", choices=("fixture", "export", "manual", "live"), default="fixture")
    run.add_argument("--n", type=int, default=10)
    run.add_argument("--config", default="config.yaml")
    run.add_argument("--anonymise", action="store_true")
    run.add_argument("--i-understand-tos", action="store_true")
    run.add_argument("--urls", default="urls.txt")
    args = parser.parse_args()
    if args.source == "live" and not args.i_understand_tos:
        parser.error("live source requires --i-understand-tos")
    from lgi.pipeline import run_pipeline
    run_pipeline(args)


if __name__ == "__main__":
    main()
