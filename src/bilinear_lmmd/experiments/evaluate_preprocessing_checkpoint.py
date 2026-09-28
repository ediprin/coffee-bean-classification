from __future__ import annotations

import argparse
import json
from pathlib import Path

from bilinear_lmmd.core.reproducibility import sha256_file
from bilinear_lmmd.engine.preprocessing_study import evaluate_preprocessing_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-only preprocessing checkpoint evaluator")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    metrics = evaluate_preprocessing_checkpoint(
        args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "split": args.split,
                "macro_f1": metrics["macro_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
