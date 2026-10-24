#!/usr/bin/env python3
"""Download and verify a CROP artifact manifest without overwriting files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote
from urllib.request import urlopen


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_destination(root: Path, relative_text: str) -> Path:
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path: {relative_text!r}")
    return root / relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/paper-v1.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--base-url", help="Override the base URL stored in the manifest.")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.root.resolve()
    base_url = args.base_url or manifest.get("base_url")
    missing = 0
    for record in manifest["files"]:
        destination = safe_destination(root, record["path"])
        if destination.exists():
            if destination.stat().st_size != record["bytes"] or sha256(destination) != record["sha256"]:
                raise SystemExit(f"Refusing to replace differing artifact: {destination}")
            print(f"ready: {record['path']}")
            continue
        missing += 1
        if args.verify_only:
            print(f"missing: {record['path']}")
            continue
        if not base_url:
            raise SystemExit("Manifest has no base_url; pass --base-url after uploading the artifacts")
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = base_url.rstrip("/") + "/" + quote(record["path"], safe="/")
        with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            with urlopen(url) as response:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        if temporary.stat().st_size != record["bytes"] or sha256(temporary) != record["sha256"]:
            temporary.unlink(missing_ok=True)
            raise SystemExit(f"Checksum mismatch for {record['path']}")
        os.replace(temporary, destination)
        print(f"downloaded: {record['path']}")
    if args.verify_only and missing:
        raise SystemExit(f"{missing} artifact(s) missing")


if __name__ == "__main__":
    main()
