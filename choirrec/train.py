"""CLI entry point: `python -m choirrec.train --small`"""

from __future__ import annotations

import argparse

from .config import ChoirRecConfig
from .pipeline import ChoirRecPipeline


def main():
    parser = argparse.ArgumentParser(description="Train ChoirRec on synthetic data.")
    parser.add_argument("--small", action="store_true", help="Use the tiny smoke-test config.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--llm-backend", choices=["mock", "qwen"], default=None)
    args = parser.parse_args()

    cfg = ChoirRecConfig().small() if args.small else ChoirRecConfig()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.llm_backend is not None:
        cfg.llm.backend = args.llm_backend

    metrics = ChoirRecPipeline(cfg).run()
    print("\n=== Final test metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
