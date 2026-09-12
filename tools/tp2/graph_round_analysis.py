import sqlite3
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r'D:\AI_envs\aider_olymp\project_ninfer\logs\mtp2_graph_prof.sqlite'
db = sqlite3.connect(path)
cur = db.cursor()

tables = [t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('tables of interest:',
      [t for t in tables if 'CUPTI' in t or 'GPU' in t.upper()])

# Syncs => round cadence.
sync = cur.execute('SELECT start, end FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION ORDER BY start'
                   ).fetchall()
print(f'\nsync rows: {len(sync)}')
if len(sync) > 10:
    tail = sync[-40:]
    gaps = sorted((tail[i+1][0] - tail[i][0]) / 1e6 for i in range(len(tail)-1))
    print(f'round wall median {gaps[len(gaps)//2]:.2f}ms  min {gaps[0]:.2f}  max {gaps[-1]:.2f}')

# Memcpy traffic: what is still staged inside the graphed round?
mc = cur.execute('SELECT start, end, deviceId, streamId, bytes, copyKind, graphNodeId '
                 'FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start').fetchall()
print(f'\nmemcpy rows: {len(mc)}')
sized = {}
for m in mc:
    b = m[4] or 0
    key = b if b else 'unknown'
    if key not in sized:
        sized[key] = [0, 0]
    sized[key][0] += 1
    sized[key][1] += (m[1] - m[0])
print('size -> count, summed us (top by total bytes):')
for b, (cnt, dur) in sorted(sized.items(), key=lambda kv: -(kv[0] if isinstance(kv[0], int) else 0) * kv[1][0])[:12]:
    bs = f'{b/1024:.1f}KiB' if isinstance(b, int) else b
    print(f'  {bs:>12}  n={cnt:7d}  sum={dur/1000:9.1f}ms  avg={dur/cnt/1000:7.1f}us')

if len(sync) > 10:
    s0 = sync[-20][0]
    s1 = sync[-19][0]
    window = [m for m in mc if s0 <= m[0] < s1]
    tot = sum(m[4] or 0 for m in window)
    dur = sum(m[1]-m[0] for m in window)
    print(f'\none round ({(s1-s0)/1e6:.2f}ms): {len(window)} memcpys, '
          f'{tot/1e6:.2f}MB staged, summed {dur/1000:.2f}ms')
    top = sorted(window, key=lambda m: -(m[1]-m[0]))[:6]
    for m in top:
        print(f'  {(m[4] or 0)/1024:9.1f}KiB  {(m[1]-m[0])/1000:8.1f}us dev{m[2]} '
              f'graphNode={m[6]}')
