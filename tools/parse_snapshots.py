#!/usr/bin/env python3
"""
tools/parse_snapshots.py

Splits a `-mmlir --mlir-print-ir-after-all` stream (as produced by clang with
CIR enabled) into one .mlir file per pass, and writes a manifest.json
describing the ordered chain of snapshots.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HEADER_RE = re.compile(
    r"^// -----// IR Dump After "
    r"(?P<pass_class>[\w:]+)"
    r":\s*"
    r"(?P<pass_name>[\w\-]+)"
    r"\s*\("
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slug(pass_name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", pass_name)


def split_stream(stream_text: str):
    lines = stream_text.splitlines(keepends=True)
    header_indices = [i for i, ln in enumerate(lines) if HEADER_RE.match(ln)]
    if not header_indices:
        return
    header_indices.append(len(lines))
    for k in range(len(header_indices) - 1):
        header_line_idx = header_indices[k]
        next_header_idx = header_indices[k + 1]
        m = HEADER_RE.match(lines[header_line_idx])
        if not m:
            continue
        pass_class = m.group("pass_class")
        pass_name = m.group("pass_name")
        body = "".join(lines[header_line_idx + 1 : next_header_idx]).rstrip() + "\n"
        yield pass_class, pass_name, body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream", required=True, type=Path,
                    help="Path to snapshot-stream.txt from clang.")
    ap.add_argument("--outdir", required=True, type=Path,
                    help="Directory for per-pass .mlir files and manifest.json.")
    ap.add_argument("--source", type=Path, default=None,
                    help="Original source file, recorded in manifest for provenance.")
    args = ap.parse_args()

    if not args.stream.is_file():
        print(f"error: stream file not found: {args.stream}", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)
    stream_text = args.stream.read_text(encoding="utf-8", errors="replace")

    snapshots = []
    parent_hash = None

    for idx, (pass_class, pass_name, body) in enumerate(split_stream(stream_text)):
        filename = f"{idx:02d}_{slug(pass_name)}.mlir"
        outpath = args.outdir / filename
        outpath.write_text(body, encoding="utf-8")
        body_hash = sha256(body)
        snapshots.append({
            "index": idx,
            "pass_class": pass_class,
            "pass_name": pass_name,
            "filename": filename,
            "bytes": len(body.encode("utf-8")),
            "lines": body.count("\n"),
            "sha256": body_hash,
            "parent_sha256": parent_hash,
        })
        parent_hash = body_hash

    manifest = {
        "source": str(args.source) if args.source else None,
        "source_sha256": sha256(args.source.read_text(errors="replace"))
                         if args.source and args.source.is_file() else None,
        "stream_file": str(args.stream),
        "stream_sha256": sha256(stream_text),
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
    }

    manifest_path = args.outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Parsed {len(snapshots)} snapshots into {args.outdir}")
    for s in snapshots:
        print(f"  {s['index']:02d}  {s['pass_name']:<28s}  "
              f"{s['lines']:>6d} lines  {s['bytes']:>7d} B  {s['sha256'][:8]}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
