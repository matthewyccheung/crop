"""Export targeted CROP subsets for cheap verifier baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crop.cheap_baselines import (
    export_coe_jsonl,
    export_text_npz,
    select_target_traces,
    summarize_npz,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_root", default="data/crop_hf")
    parser.add_argument("--output_dir", default="data/cheap_baselines")
    parser.add_argument("--dataset_prefix", default="crop_target")
    parser.add_argument("--arithmetic_n", type=int, default=1500)
    parser.add_argument("--boolean_n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=2806)
    parser.add_argument("--smoke_n", type=int, default=12)
    args = parser.parse_args()

    out = Path(args.output_dir)
    traces = select_target_traces(
        args.crop_root,
        arithmetic_n=args.arithmetic_n,
        boolean_n=args.boolean_n,
        include_all_gsm8k=True,
        seed=args.seed,
    )
    coe_paths = export_coe_jsonl(traces, out / "coe_jsonl", dataset_prefix=args.dataset_prefix)
    text_step = export_text_npz(traces, out / f"{args.dataset_prefix}_text_steps.npz", granularity="step")
    text_trace = export_text_npz(traces, out / f"{args.dataset_prefix}_text_traces.npz", granularity="trace")

    smoke = traces[: args.smoke_n]
    if smoke:
        export_coe_jsonl(smoke, out / "coe_jsonl", dataset_prefix=f"{args.dataset_prefix}_smoke")
        export_text_npz(smoke, out / f"{args.dataset_prefix}_smoke_text_steps.npz", granularity="step")
        export_text_npz(smoke, out / f"{args.dataset_prefix}_smoke_text_traces.npz", granularity="trace")

    rows = [summarize_npz(text_step), summarize_npz(text_trace)]
    pd.DataFrame(rows).to_csv(out / f"{args.dataset_prefix}_dataset_summary.csv", index=False)
    print(f"selected_traces={len(traces)}")
    for name, path in coe_paths.items():
        print(f"{name}={path}")
    print(f"text_step={text_step}")
    print(f"text_trace={text_trace}")


if __name__ == "__main__":
    main()
