"""Shard CoE JSONL datasets and manifests for parallel GPU scoring."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--n_shards", type=int, default=4)
    args = parser.parse_args()

    items = read_jsonl(Path(args.jsonl))
    rows = read_manifest(Path(args.manifest))
    if len(items) != len(rows):
        raise ValueError(f"jsonl rows {len(items)} != manifest rows {len(rows)}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    for shard in range(args.n_shards):
        shard_items = []
        shard_rows = []
        dataset = f"{args.prefix}_s{shard:02d}"
        for local_idx, global_idx in enumerate(range(shard, len(items), args.n_shards)):
            item = dict(items[global_idx])
            row = dict(rows[global_idx])
            item["id"] = local_idx
            item["dataset"] = dataset
            item["dataset_index"] = local_idx
            row["dataset"] = dataset
            row["dataset_index"] = local_idx
            row["global_index"] = global_idx
            if "global_index" not in fieldnames:
                fieldnames.append("global_index")
            shard_items.append(item)
            shard_rows.append(row)
        jsonl_path = out / f"{dataset}.jsonl"
        manifest_path = out / f"{dataset}_manifest.csv"
        with jsonl_path.open("w") as f:
            for item in shard_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(shard_rows)
        print(f"{dataset}: rows={len(shard_items)} jsonl={jsonl_path} manifest={manifest_path}")


if __name__ == "__main__":
    main()
