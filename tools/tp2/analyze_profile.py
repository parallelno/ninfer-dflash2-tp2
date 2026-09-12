"""Analyze an nsys SQLite export: table census, kernel/memcpy rows, decode-window breakdown."""
import sqlite3
import sys

path = sys.argv[1]
con = sqlite3.connect(path)
cur = con.cursor()
tabs = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
interesting = [t for t in tabs if any(k in t.upper() for k in ("KERNEL", "GRAPH", "MEMCPY"))]
print(f"tables: {len(tabs)}")
for t in interesting:
    n = cur.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    print(f"  {t}: {n} rows")

if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs:
    print("\n=== memcpy kind distribution ===")
    for row in cur.execute(
        "SELECT copyKind, COUNT(*), SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
    ):
        print(f"  kind {row[0]}: {row[1]:>7} copies, {row[2]:>12} B")
    print("\n=== memcpy byte distribution ===")
    for row in cur.execute(
        "SELECT bytes, copyKind, COUNT(*) c FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY bytes, copyKind ORDER BY c DESC LIMIT 12"
    ):
        print(f"  {row[0]:>12} bytes kind={row[1]} x {row[2]:>7}")
    print("\n=== memcpy rate over time (1s buckets) ===")
    t0 = cur.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()[0]
    for row in cur.execute(
        "SELECT (start - ?) / 1000000000, COUNT(*), SUM(bytes), SUM(end-start)/1000000.0 "
        "FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY 1 ORDER BY 1",
        (t0,),
    ):
        print(f"  t+{row[0]:>3}s: {row[1]:>7} copies, {row[2]:>12} B, {row[3]:>9.1f} ms busy")

    # Inter-op gaps: are the two dominant packet sizes serialized back-to-back?
    print("\n=== inter-copy gap distribution (4352B and 3072B kinds) ===")
    rows = cur.execute(
        "SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY "
        "WHERE bytes IN (4352, 3072) ORDER BY start"
    ).fetchall()
    gaps = []
    for i in range(1, len(rows)):
        gaps.append(rows[i][0] - rows[i - 1][1])
    gaps.sort()
    n = len(gaps)
    for label, v in (
        ("p10", gaps[n // 10]),
        ("p50", gaps[n // 2]),
        ("p90", gaps[9 * n // 10]),
        ("p99", gaps[99 * n // 100]),
    ):
        print(f"  gap {label}: {v / 1000.0:.2f} us")

if "CUPTI_ACTIVITY_KIND_RUNTIME" in tabs or "CUPTI_ACTIVITY_KIND_DRIVER" in tabs:
    src = (
        "CUPTI_ACTIVITY_KIND_RUNTIME"
        if "CUPTI_ACTIVITY_KIND_RUNTIME" in tabs
        else "CUPTI_ACTIVITY_KIND_DRIVER"
    )
    print(f"\n=== API rows in {src} ===")
    for row in cur.execute(
        f'SELECT nameId, COUNT(*) FROM "{src}" GROUP BY nameId ORDER BY COUNT(*) DESC LIMIT 10'
    ):
        name = cur.execute(
            "SELECT value FROM StringIds WHERE id = ?", (row[0],)
        ).fetchone()
        print(f"  {name[0] if name else row[0]}: {row[1]}")


