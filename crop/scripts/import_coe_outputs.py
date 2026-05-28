"""Import Chain-of-Embedding outputs as score-column NPZ files."""

from __future__ import annotations

import argparse

from crop.cheap_baselines import import_coe_outputs, summarize_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--language", default="en")
    parser.add_argument("--token_aggregation", default="average")
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()
    path = import_coe_outputs(
        project_path=args.project_path,
        manifest_path=args.manifest,
        output_npz=args.output,
        model_name=args.model_name,
        language=args.language,
        token_aggregation=args.token_aggregation,
        strict=not args.allow_missing,
    )
    print(summarize_npz(path))


if __name__ == "__main__":
    main()
