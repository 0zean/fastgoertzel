"""Render the README benchmark tables from a `benchmark.py --json` report.

    python benchmarks/render_tables.py bench.json                 # print markdown
    python benchmarks/render_tables.py bench.json --update README.md

With --update, the text between the markers

    <!-- benchmark-tables:start -->
    <!-- benchmark-tables:end -->

in the README is replaced, so the published numbers are always the ones in the report and
carry the machine description from the same run. Prose around the tables is left alone.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

START = "<!-- benchmark-tables:start -->"
END = "<!-- benchmark-tables:end -->"


def fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def fmt_n(n: int) -> str:
    return f"{n:,}"


def ratio(ref: float, mine: float) -> str:
    return f"{ref / mine:.1f}x"


def render(report: dict[str, Any]) -> str:
    env = report["environment"]
    out: list[str] = []
    out.append(
        f"Measured with `python benchmarks/benchmark.py` on {env['cpu']} "
        f"({env['logical_cores']} logical cores), {env['platform']}, Python {env['python']}, "
        f"NumPy {env['numpy']}, fastgoertzel {env['fastgoertzel']}"
        f"{', ' + env['kernel'] + ' kernel' if env.get('kernel') else ''}. "
        "Each time is the minimum of the timing rounds; ratios are baseline time over "
        "fastgoertzel time."
    )

    if "single" in report:
        out.append("")
        out.append("**One frequency**, `goertzel(x, f)`, single-threaded:")
        out.append("")
        out.append(
            "| N | fastgoertzel | pure-Python loop | NumPy 2x real dot | `np.fft.rfft`, one bin |"
        )
        out.append("|---:|---:|---:|---:|---:|")
        for row in report["single"]:
            fg = row["fastgoertzel"]["seconds_min"]
            py = row.get("pure_python")
            py_cell = (
                f"{fmt_time(py['seconds_min'])} ({ratio(py['seconds_min'], fg)})"
                if py
                else "not run"
            )
            nd = row["numpy_two_dots"]["seconds_min"]
            ff = row["numpy_rfft"]["seconds_min"]
            out.append(
                f"| {fmt_n(row['n'])} | {fmt_time(fg)} | {py_cell} | "
                f"{fmt_time(nd)} ({ratio(nd, fg)}) | {fmt_time(ff)} ({ratio(ff, fg)}) |"
            )

    if "batch" in report:
        out.append("")
        out.append("**Several frequencies**, `goertzel_batch(x, freqs)` (OpenMP default threads):")
        out.append("")
        out.append(
            "| N | M | fastgoertzel | M x `goertzel` | `np.fft.rfft` (all bins) | M x NumPy dot |"
        )
        out.append("|---:|---:|---:|---:|---:|---:|")
        for row in report["batch"]:
            b = row["fastgoertzel_batch"]["seconds_min"]
            s = row["m_x_goertzel"]["seconds_min"]
            ff = row["numpy_rfft_all_bins"]["seconds_min"]
            nd = row["m_x_numpy_two_dots"]["seconds_min"]
            out.append(
                f"| {fmt_n(row['n'])} | {row['m']} | {fmt_time(b)} | {fmt_time(s)} ({ratio(s, b)}) | "
                f"{fmt_time(ff)} ({ratio(ff, b)}) | {fmt_time(nd)} ({ratio(nd, b)}) |"
            )

    if "sliding" in report:
        out.append("")
        out.append(
            "**Sliding windows**, `goertzel_sliding_batch(x, W, freqs)` (OpenMP default threads):"
        )
        out.append("")
        out.append("| N | W | M | fastgoertzel | NumPy `sliding_window_view` + `rfft` |")
        out.append("|---:|---:|---:|---:|---:|")
        for row in report["sliding"]:
            s = row["fastgoertzel_sliding"]["seconds_min"]
            n = row["numpy_sliding_rfft"]["seconds_min"]
            out.append(
                f"| {fmt_n(row['n'])} | {row['w']} | {row['m']} | {fmt_time(s)} | "
                f"{fmt_time(n)} ({ratio(n, s)}) |"
            )

    if report.get("threads"):
        out.append("")
        out.append("**OpenMP scaling**, batch N = 1e6, M = 32 and sliding N = 1e5, W = 256, M = 8:")
        out.append("")
        out.append("| OMP_NUM_THREADS | batch | sliding |")
        out.append("|---:|---:|---:|")
        for row in report["threads"]:
            out.append(
                f"| {row['threads']} | {fmt_time(row['batch_seconds'])} | {fmt_time(row['sliding_seconds'])} |"
            )

    if "latency" in report:
        calls = report["latency"][0]["calls"] if report["latency"] else 0
        out.append("")
        out.append(f"**Per-call latency**, {calls:,} calls each, Python overhead included:")
        out.append("")
        out.append("| Call | p50 | p90 | p99 | p99.9 | max |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for row in report["latency"]:
            out.append(
                f"| {row['case']} | {row['p50_us']:.1f} us | {row['p90_us']:.1f} us | "
                f"{row['p99_us']:.1f} us | {row['p999_us']:.1f} us | {row['max_us']:.0f} us |"
            )
    return "\n".join(out) + "\n"


def update_readme(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        sys.exit(f"{path}: markers {START} / {END} not found")
    new = pattern.sub(lambda _: f"{START}\n{block}{END}", text)
    path.write_text(new, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("report", type=Path, help="JSON written by benchmark.py --json")
    ap.add_argument("--update", type=Path, default=None, help="README to update in place")
    args = ap.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    block = render(report)
    if args.update:
        update_readme(args.update, block)
        print(f"updated {args.update}")
    else:
        print(block, end="")


if __name__ == "__main__":
    main()
