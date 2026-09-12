import sqlite3
import sys

db = sqlite3.connect(r'D:\AI_envs\aider_olymp\project_ninfer\logs\mtp2_eager_prof.sqlite')
cur = db.cursor()
tables = [t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
kern = [t for t in tables if 'KERNEL' in t.upper()]
print('kernel tables:', kern)
if not kern:
    print('all tables:', tables)
    sys.exit(1)
t = kern[0]
n = cur.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
print('rows:', n)
rows = cur.execute(
    'SELECT s.value, COUNT(*), AVG(k.end-k.start), SUM(k.end-k.start) FROM {t} k '
    'JOIN StringIds s ON k.demangledName = s.id '
    'GROUP BY k.demangledName ORDER BY SUM(k.end-k.start) DESC LIMIT 30'.format(t=t)).fetchall()
total = 0
for name, cnt, avg, tot in rows:
    total += tot
    print(f'{cnt:8d}  avg={avg/1000:8.1f}us  tot={tot/1e6:9.1f}ms  {name[:90]}')
print(f'(top30 total {total/1e6:.1f}ms)')
