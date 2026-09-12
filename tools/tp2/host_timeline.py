import sqlite3

db = sqlite3.connect(r'D:\AI_envs\aider_olymp\project_ninfer\logs\mtp2_eager_prof.sqlite')
cur = db.cursor()

# Map API name ids.
api = {}
for i, v in cur.execute('SELECT id, value FROM StringIds').fetchall():
    api[i] = v

rt = cur.execute(
    'SELECT start, end, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME '
    'ORDER BY start').fetchall()
print(f'runtime API rows: {len(rt)}')
counts = {}
durs = {}
for s, e, c, n in rt:
    name = api.get(n, str(n))
    counts[name] = counts.get(name, 0) + 1
    durs[name] = durs.get(name, 0) + (e - s)
print('\n== API call counts / total host-side duration (us) ==')
for name, cnt in sorted(counts.items(), key=lambda kv: -kv[1])[:25]:
    print(f'{cnt:8d}  {durs[name]/1000:10.1f}ms  {name[:60]}')

# Host syncs: when does the host block?
sync = cur.execute(
    'SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION ORDER BY start'
).fetchall()
print(f'\nsync rows: {len(sync)}')
if sync:
    # Inter-sync intervals => round times. Focus on the tail (decode), skip load/prefill.
    tail = sync[-60:] if len(sync) > 60 else sync
    gaps = [(tail[i+1][0] - tail[i][1]) / 1e6 for i in range(len(tail)-1)]
    blocks = [(e - s) / 1e6 for s, e, _ in tail]
    gaps_sorted = sorted(gaps)
    blocks_sorted = sorted(blocks)
    med = lambda a: a[len(a)//2] if a else 0
    print(f'tail syncs: {len(tail)}')
    print(f'round wall (sync-to-sync): median {med(gaps_sorted):.2f}ms '
          f'min {gaps_sorted[0]:.2f} max {gaps_sorted[-1]:.2f}')
    print(f'host blocked in sync: median {med(blocks_sorted):.2f}ms '
          f'min {blocks_sorted[0]:.2f} max {blocks_sorted[-1]:.2f}ms')

# What the host submits within one round: API calls between two late syncs.
if len(sync) >= 40:
    s0 = sync[-30][1]
    s1 = sync[-29][1]
    in_round = [r for r in rt if s0 <= r[0] < s1]
    sub = {}
    for s, e, c, n in in_round:
        name = api.get(n, str(n))
        sub[name] = sub.get(name, 0) + 1
    print(f'\n== one round window ({(s1-s0)/1e6:.2f}ms): {len(in_round)} API calls ==')
    for name, cnt in sorted(sub.items(), key=lambda kv: -kv[1])[:20]:
        print(f'{cnt:6d}  {name[:60]}')

# Memcpy traffic within that round window.
cols = [c[1] for c in cur.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_MEMCPY)').fetchall()]
print('\nmemcpy cols:', cols)
has_bytes = 'bytes' in cols
if has_bytes:
    mc = cur.execute('SELECT start, end, deviceId, streamId, bytes, copyKind FROM '
                     'CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start').fetchall()
else:
    mc = cur.execute('SELECT start, end, deviceId, streamId FROM '
                     'CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start').fetchall()
print(f'memcpy rows total: {len(mc)}')
if len(sync) >= 40 and mc:
    in_round_mc = [m for m in mc if s0 <= m[0] < s1]
    dur = sum(m[1]-m[0] for m in in_round_mc)
    print(f'round memcpys: {len(in_round_mc)} summed duration {dur/1000:.2f}ms')
    big = sorted(in_round_mc, key=lambda m: -(m[1]-m[0]))[:10]
    for m in big:
        if has_bytes:
            print(f'  {m[4]/1024:9.1f}KiB  {(m[1]-m[0])/1000:8.1f}us dev{m[2]} kind{m[5]}')
        else:
            print(f'  {(m[1]-m[0])/1000:8.1f}us dev{m[2]}')
