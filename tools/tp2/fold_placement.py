import sqlite3

db = sqlite3.connect(r'D:\AI_envs\aider_olymp\project_ninfer\logs\mtp2_decode_prof.sqlite')
cur = db.cursor()

gt = cur.execute('SELECT start, end, deviceId, streamId, graphId FROM '
                'CUPTI_ACTIVITY_KIND_GRAPH_TRACE ORDER BY start').fetchall()
by_dev = {}
for s, e, d, st, g in gt:
    by_dev.setdefault(d, []).append((s, e))
print('graph launches per device:')
for d, rows in by_dev.items():
    walls = sorted((e - s)/1e6 for s, e in rows)
    print(f'  dev{d}: {len(rows)} launches, median wall {walls[len(walls)//2]:.2f}ms')

if len(by_dev) == 2:
    d0 = sorted(by_dev[0])
    d1 = sorted(by_dev[1])
    pairs = 0
    for s0, e0 in d0[:50]:
        for s1, e1 in d1:
            if s1 > e0:
                if s1 < e0 + 5e6:
                    pairs += 1
                break
    print(f'dev0 graphs immediately followed by dev1 graph within 5ms: {pairs}/{min(len(d0),50)}')

k = cur.execute('SELECT start, end, deviceId, graphNodeId, graphId, launchType FROM '
                'CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
in_graph = sum(1 for r in k if r[5] == 1)
print(f'\nfold kernels: {len(k)}, launchType==1 (graph-node): {in_graph}')

folds = sorted((r[0], r[1], r[2]) for r in k)
for d in sorted(by_dev):
    graphs = sorted(by_dev[d])
    inside = []
    after = []
    for fs, fe, fd in folds:
        if fd != d:
            continue
        placed = False
        for gs, ge in graphs:
            if gs <= fs <= ge:
                inside.append((fs - gs)/1e6)
                placed = True
                break
        if not placed:
            prev = [g for g in graphs if g[1] < fs]
            if prev and fs - prev[-1][1] < 5e6:
                after.append((fs - prev[-1][1])/1e6)
    print(f'dev{d}: folds inside graph window {len(inside)} '
          f'(latest at {max(inside) if inside else -1:.2f}ms); '
          f'after graph end {len(after)} '
          f'(median offset {sorted(after)[len(after)//2] if after else -1:.2f}ms)')

d0 = sorted(by_dev[0]) if 0 in by_dev else []
if len(d0) > 10:
    busy = sum(e-s for s, e in d0)/1e6
    span = (d0[-1][1]-d0[0][0])/1e6
    print(f'\ndev0 graph busy {busy:.2f}ms of {span:.2f}ms span')
