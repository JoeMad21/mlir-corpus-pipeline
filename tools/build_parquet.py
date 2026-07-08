#!/usr/bin/env python3
"""
tools/build_parquet.py

Aggregates the loose per-benchmark corpus tree into a Hive-partitioned
Parquet dataset. One row per surviving snapshot.

Policies (locked in from design discussion):
    Whales:         Cap ir_text at 2 MB; mark truncated=True.
    Dedup:          Keep most-recent run per (benchmark_name, pass_name).
    Failed rounds:  Skip snapshots where roundtrip_status='failed'.
    Partitioning:   Hive-partition by benchmark_name.

Usage:
    python3 tools/build_parquet.py \\
        --input /mnt/nvme10/joseph_ufl/corpus-alpha01 \\
        --outdir /mnt/nvme10/joseph_ufl/corpus-parquet
"""

import argparse
import datetime
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False


DIALECT_RE = re.compile(r'\b([a-z_][a-z0-9_]*)\.[a-z_][a-z0-9_]+')


def extract_dialects(ir_text, cap=200_000):
    seen = set()
    for m in DIALECT_RE.finditer(ir_text[:cap]):
        seen.add(m.group(1))
    return sorted(seen)


def sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parse_run_dir_name(name):
    m = re.match(r'^(.+)-(\d{8}-\d{6})$', name)
    if not m:
        return None
    return m.group(1), m.group(2)


def collect_snapshots(input_root, source_lang_tag):
    for run_dir in sorted(input_root.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_dir_name(run_dir.name)
        if parsed is None:
            continue
        benchmark_name, run_timestamp = parsed

        manifest_path = run_dir / 'snapshots' / 'manifest.json'
        if not manifest_path.is_file():
            continue

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f'WARN: could not read {manifest_path}: {e}', file=sys.stderr)
            continue

        source_sha256 = manifest.get('source_sha256', '')

        for snap in manifest.get('snapshots', []):
            if snap.get('roundtrip_status') != 'ok':
                continue

            snap_file = run_dir / 'snapshots' / snap['filename']
            if not snap_file.is_file():
                continue

            yield {
                'run_dir': run_dir,
                'run_timestamp': run_timestamp,
                'benchmark_name': benchmark_name,
                'source_lang': source_lang_tag,
                'pass_index': snap['index'],
                'pass_class': snap['pass_class'],
                'pass_name': snap['pass_name'],
                'snapshot_path': snap_file,
                'ir_sha256_from_manifest': snap.get('sha256', ''),
                'parent_sha256': snap.get('parent_sha256'),
                'source_sha256': source_sha256,
            }


def dedupe_by_bench_pass(rows):
    latest = {}
    for row in rows:
        key = (row['benchmark_name'], row['pass_name'])
        prev = latest.get(key)
        if prev is None or row['run_timestamp'] > prev['run_timestamp']:
            latest[key] = row
    result = list(latest.values())
    result.sort(key=lambda r: (r['benchmark_name'], r['pass_index']))
    return result


def build_row(entry, cap_bytes, ingest_utc):
    raw = entry['snapshot_path'].read_text(encoding='utf-8', errors='replace')
    raw_bytes = raw.encode('utf-8')
    ir_bytes = len(raw_bytes)
    ir_lines = raw.count('\n')

    if ir_bytes > cap_bytes:
        cut = cap_bytes
        while cut > 0 and cut < len(raw_bytes) and (raw_bytes[cut] & 0xC0) == 0x80:
            cut -= 1
        ir_text = raw_bytes[:cut].decode('utf-8', errors='replace')
        truncated = True
    else:
        ir_text = raw
        truncated = False

    return {
        'benchmark_name': entry['benchmark_name'],
        'source_lang': entry['source_lang'],
        'pass_index': int(entry['pass_index']),
        'pass_class': entry['pass_class'],
        'pass_name': entry['pass_name'],
        'ir_text': ir_text,
        'ir_bytes': ir_bytes,
        'ir_lines': ir_lines,
        'ir_sha256': sha256(raw),
        'parent_sha256': entry['parent_sha256'],
        'truncated': truncated,
        'dialects_present': extract_dialects(raw),
        'source_sha256': entry['source_sha256'],
        'run_timestamp': entry['run_timestamp'],
        'corpus_ingest_utc': ingest_utc,
    }


_SCHEMA = None
def get_schema():
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = pa.schema([
            ('benchmark_name',    pa.string()),
            ('source_lang',       pa.string()),
            ('pass_index',        pa.int32()),
            ('pass_class',        pa.string()),
            ('pass_name',         pa.string()),
            ('ir_text',           pa.large_string()),
            ('ir_bytes',          pa.int64()),
            ('ir_lines',          pa.int32()),
            ('ir_sha256',         pa.string()),
            ('parent_sha256',     pa.string()),
            ('truncated',         pa.bool_()),
            ('dialects_present',  pa.list_(pa.string())),
            ('source_sha256',     pa.string()),
            ('run_timestamp',     pa.string()),
            ('corpus_ingest_utc', pa.string()),
        ])
    return _SCHEMA


def write_partition(rows, outdir, benchmark_name):
    part_dir = outdir / f'benchmark_name={benchmark_name}'
    part_dir.mkdir(parents=True, exist_ok=True)
    out_path = part_dir / 'shard-00000.parquet'

    payload = [{k: v for k, v in r.items() if k != 'benchmark_name'} for r in rows]
    schema = pa.schema([f for f in get_schema() if f.name != 'benchmark_name'])
    table = pa.Table.from_pylist(payload, schema=schema)

    pq.write_table(
        table,
        out_path,
        compression='zstd',
        compression_level=3,
        row_group_size=512,
    )
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--input',       required=True, type=Path)
    ap.add_argument('--outdir',      required=True, type=Path)
    ap.add_argument('--source-lang', default='cuda-host')
    ap.add_argument('--cap-bytes',   default=2 * 1024 * 1024, type=int)
    ap.add_argument('--dry-run',     action='store_true')
    args = ap.parse_args()

    if not args.input.is_dir():
        print(f'error: --input is not a directory: {args.input}', file=sys.stderr)
        return 2

    print(f'[build_parquet] input:       {args.input}')
    print(f'[build_parquet] outdir:      {args.outdir}')
    print(f'[build_parquet] source_lang: {args.source_lang}')
    print(f'[build_parquet] cap_bytes:   {args.cap_bytes}')
    print(f'[build_parquet] dry_run:     {args.dry_run}')
    print()

    print('[1/3] Walking input directories...')
    all_entries = list(collect_snapshots(args.input, args.source_lang))
    print(f'    Found {len(all_entries)} candidate snapshots.')

    print('[2/3] Deduplicating by (benchmark_name, pass_name)...')
    deduped = dedupe_by_bench_pass(all_entries)
    print(f'    {len(deduped)} unique snapshots after dedup '
          f'({len(all_entries) - len(deduped)} duplicates dropped).')

    if not deduped:
        print('error: no snapshots to write.', file=sys.stderr)
        return 3

    print('[3/3] Reading IR text, capping whales, grouping by benchmark...')
    ingest_utc = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    by_bench = defaultdict(list)
    truncated_count = 0
    total_bytes_before = 0
    total_bytes_after = 0

    for entry in deduped:
        row = build_row(entry, args.cap_bytes, ingest_utc)
        by_bench[row['benchmark_name']].append(row)
        total_bytes_before += row['ir_bytes']
        total_bytes_after  += len(row['ir_text'].encode('utf-8'))
        if row['truncated']:
            truncated_count += 1

    print(f'    {len(deduped)} rows built.')
    print(f'    {truncated_count} truncated by cap.')
    print(f'    Original text bytes:  {total_bytes_before:,}')
    print(f'    Capped text bytes:    {total_bytes_after:,}')

    dialect_counts = defaultdict(int)
    for rows in by_bench.values():
        for row in rows:
            for d in row['dialects_present']:
                dialect_counts[d] += 1
    print(f'    Distinct dialects observed: {len(dialect_counts)}')
    top = sorted(dialect_counts.items(), key=lambda x: -x[1])[:12]
    for name, count in top:
        print(f'      {count:>6d}  {name}')

    if args.dry_run:
        print()
        print('DRY RUN: skipping Parquet write.')
        print(f'Would write {len(by_bench)} partition directories under {args.outdir}.')
        return 0

    if not HAVE_PYARROW:
        print('error: pyarrow is required. Install with: '
              'python3 -m pip install --user pyarrow',
              file=sys.stderr)
        return 4

    args.outdir.mkdir(parents=True, exist_ok=True)
    written = 0
    for bench, rows in sorted(by_bench.items()):
        path = write_partition(rows, args.outdir, bench)
        written += 1
        print(f'    {bench:<40s}  {len(rows):>2d} rows  {path.stat().st_size / 1024:>7.1f} KiB')

    print()
    print(f'[build_parquet] Wrote {written} partitions to {args.outdir}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
