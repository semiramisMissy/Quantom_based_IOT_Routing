"""One command: python run_all.py [n_seeds]   (default 10; use 3 for a ~20 s quick run)
Runs simulation -> tables/figures -> sample network. All outputs are written next to the scripts."""
import sys, io, contextlib, runpy, time
import pandas as pd, morq_sim as m
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
t0 = time.time(); rows = []
for v in [2, 5, 10, 15, 20]:
    for sd in range(n):
        for p in m.PROTOS: rows.append(m.simulate(p, sd, v))
    print(f'speed {v} m/s done ({time.time()-t0:.0f}s)', flush=True)
pd.DataFrame(rows).to_csv('raw_results.csv', index=False)
runpy.run_path('analyze.py')
buf = io.StringIO()
with contextlib.redirect_stdout(buf): runpy.run_path('sample_run.py')
open('sample_run_output.txt', 'w').write(buf.getvalue()); print('\n' + buf.getvalue())
print(f'Done in {time.time()-t0:.0f}s. Files: raw_results.csv, table_overall.csv, paired_tests.csv, fig1..fig6 (png)')
