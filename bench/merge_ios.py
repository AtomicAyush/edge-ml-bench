"""Add an iPhone benchmark export (EdgeDetect's Benchmark tab -> Share) to the results.

    python -m bench.merge_ios ~/Downloads/edge-ml-bench-results.csv

Columns shared with results/latency.csv go there, so the phone appears in the
latency chart. The full rows, with load times and end-to-end latency, go to
results/ios.csv. Rows are read with a CSV parser, so quoted device names such as
"iPhone18,2" stay in one column. Rows already present (same timestamp, device,
variant and backend) are skipped, so merging the same file twice is harmless.
"""

import csv
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
LATENCY_COLUMNS = ["timestamp", "device", "variant", "backend", "p50_ms", "p95_ms", "mean_ms", "fps", "size_mb"]
KEY = ("timestamp", "device", "variant", "backend")


def existing(path: Path, columns: list[str]) -> set[tuple]:
    """Keys already in `path`; fails before anything is written if its columns differ."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != columns:
            raise SystemExit(f"{path.name} has columns {reader.fieldnames}, expected {columns}")
        return {tuple(r[k] for k in KEY) for r in reader}


def append(path: Path, rows: list[dict], columns: list[str]):
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    with open(sys.argv[1], newline="") as fh:
        reader = csv.DictReader(fh)
        rows, columns = list(reader), reader.fieldnames or []
    missing = set(LATENCY_COLUMNS) - set(columns)
    if missing:
        raise SystemExit(f"not an EdgeDetect export: missing {sorted(missing)}")
    targets = [(RESULTS / "latency.csv", LATENCY_COLUMNS), (RESULTS / "ios.csv", columns)]
    seen = [existing(path, cols) for path, cols in targets]   # check both before writing either
    for (path, cols), keys in zip(targets, seen):
        append(path, [r for r in rows if tuple(r[k] for k in KEY) not in keys], cols)
    added = sum(tuple(r[k] for k in KEY) not in seen[0] for r in rows)
    print(f"added {added} new rows from {rows[0]['device'] if rows else '?'} ({len(rows) - added} already present)")
