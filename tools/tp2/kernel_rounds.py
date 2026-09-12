import sqlite3
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r'D:\AI_envs\aider_olymp\project_ninfer\logs\mtp2_decode_prof.sqlite'
db = sqlite3.connect(path)
cur = db.cursor()

cols = [c[1] for c in cur.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)').fetchall()]
print('kernel cols:', cols)
gcols = [c[1] for c in cur.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_GRAPH_TRACE)').fetchall()]
print('graph cols:', gcols)

api = {}
for i, v in cur.execute('SELECT id, value FROM StringIds').fetchall():
    api[i] = v

gt = cur.execute('SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE ORDER BY start').fetchall()
if gt:
    print(f'\ngraph rows: {len(gt)}')
    if len(gt) > 20:
        walls = sorted((e - s) / 1e6 for s, e in gt)
        print(f'graph exec wall: median {walls[len(walls)//2]:.2f}ms '
              f'p10 {walls[len(walls)//10]:.2f} p90 {walls[len(walls)*9//10]:.2f}')
        pairs = sorted((gt[i+1][0] - gt[i][0]) / 1e6 for i in range(len(gt)-1))
        print(f'graph period: median {pairs[len(pairs)//2]:.2f}ms')

k = cur.execute(
    'SELECT start, end, deviceId, streamId, graphNodeId, shortName '
    'FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start').fetchall()
print(f'\nkernel rows: {len(k)}')
if k:
    names = {}
    for row in k:
        nm = api.get(row[5], str(row[5]))
        if nm not in names:
            names[nm] = [0, 0, 0, 10**18]
        e = names[nm]
        e[0] += 1
        e[1] += (row[1] - row[0])
        e[2] = max(e[2], row[1] - row[0])
        e[3] = min(e[3], row[1] - row[0])
    print('name  count  total_ms  avg_us  min_us  max_us')
    for nm, (cnt, tot, mx, mn) in sorted(names.items(), key=lambda kv: -kv[1][1])[:30]:
        print(f'{cnt:7d} {tot/1e6:9.2f} {tot/cnt/1000:8.1f} {mn/1000:6.1f} {mx/1000:8.1f}  {nm[:70]}')
    span = (k[-1][1] - k[0][0]) / 1e6
    busy = sum(r[1] - r[0] for r in k) / 1e6
    d0 = [r for r in k if r[2] == 0]
    d1 = [r for r in k if r[2] == 1]
    b0 = sum(r[1]-r[0] for r in d0)/1e6
    b1 = sum(r[1]-r[0] for r in d1)/1e6
    print(f'\ntrace span {span:.2f}ms; kernel busy all {busy:.2f}ms '
          f'({100*busy/span:.1f}%), dev0 {b0:.2f}ms ({100*b0/span:.1f}%), '
          f'dev1 {b1:.2f}ms ({100*b1/span:.1f}%)')
    for label, rows in (('dev0', d0), ('dev1', d1)):
        rows = sorted(rows, key=lambda r: r[0])
        gaps = []
        prev_end = rows[0][1]
        for r in rows[1:]:
            if r[0] > prev_end:
                gaps.append(r[0] - prev_end)
            prev_end = max(prev_end, r[1])
        gaps.sort()
        if gaps:
            print(f'{label}: {len(rows)} kernels, {len(gaps)} gaps, '
                  f'sum gaps {(sum(gaps)/1e6):.2f}ms, median gap {gaps[len(gaps)//2]/1000:.1f}us, '
                  f'max gap {gaps[-1]/1000:.1f}us, p99 {gaps[int(len(gaps)*0.99)]/1000:.1f}us')
