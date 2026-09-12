"""Summarize nvidia-smi 1-second samples: per-GPU utilization/power/clocks during a run."""
import statistics as st
import sys

path = sys.argv[1]
rows = []
with open(path, encoding="utf-8") as fh:
    for line in fh:
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) != 6 or "%" not in parts[2]:
            continue
        try:
            rows.append(
                {
                    "idx": parts[1],
                    "util": int(parts[2].split()[0]),
                    "power": float(parts[3].split()[0]),
                    "sm": int(parts[4].split()[0]),
                    "mem": int(parts[5].split()[0]),
                }
            )
        except (ValueError, IndexError):
            continue

for idx in sorted({r["idx"] for r in rows}):
    g = [r for r in rows if r["idx"] == idx]
    busy = [r for r in g if r["util"] >= 20]
    print(f"GPU{idx}: n={len(g)}")
    if g:
        print(
            f"  all:  util {st.mean(r['util'] for r in g):5.1f}%  "
            f"power {st.mean(r['power'] for r in g):5.1f}W  "
            f"sm {st.mean(r['sm'] for r in g):4.0f}MHz  "
            f"mem {st.mean(r['mem'] for r in g):6.0f}MiB"
        )
    if busy:
        print(
            f"  busy: n={len(busy)} util {st.mean(r['util'] for r in busy):5.1f}%  "
            f"power {st.mean(r['power'] for r in busy):5.1f}W (max {max(r['power'] for r in busy):.0f})  "
            f"sm {st.mean(r['sm'] for r in busy):4.0f}MHz (min {min(r['sm'] for r in busy)} max {max(r['sm'] for r in busy)})"
        )
