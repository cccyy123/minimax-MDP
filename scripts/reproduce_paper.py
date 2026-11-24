"""Reproduce paper numerical values and figures from the supplied input datasets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from restore_inputs import restore

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();out=args.output.resolve()
    if out==ROOT or ROOT in out.parents or out in ROOT.parents:
        parser.error('Output must be outside the repository')
    out.mkdir(parents=True,exist_ok=False)
    restore()
    env=dict(os.environ,NUMBA_CACHE_DIR=str(out/'numba-cache'),MPLCONFIGDIR=str(out/'.matplotlib'))
    steps=[
        ['check_compiled_phase.py'],
        ['check_vectorized_dp.py'],
        ['reproduce.py','figure1','--output',str(out/'figure1')],
        ['reproduce.py','verification','--output',str(out/'verification')],
        ['verify_tables.py','--dp','--vectorized-dp','--workers',str(args.workers),'--output',str(out/'tables')],
        ['replay_prediction_figures.py','--output',str(out/'predictions')],
        ['replay_changing_cost_all.py','--workers',str(args.workers),'--output',str(out/'changing-cost')],
        ['build_replayed_figures.py','--predictions',str(out/'predictions'),'--changing-cost',str(out/'changing-cost'),'--output',str(out/'figures')],
        ['export_tables.py','--output',str(out/'exported-tables')],
    ]
    results=[]
    for index,step in enumerate(steps,1):
        command=[sys.executable,str(ROOT/'scripts'/step[0]),*step[1:]]
        log_path=out/f'{index:02d}-{Path(step[0]).stem}.log'
        print(f'[{index}/{len(steps)}] {step[0]} -> {log_path}',flush=True)
        start=time.monotonic()
        with log_path.open('w') as log:
            completed=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        results.append({'step':step,'exit_code':completed.returncode,'wall_seconds':time.monotonic()-start,'log':log_path.name})
        (out/'RUN.json').write_text(json.dumps(results,indent=2))
        if completed.returncode:raise SystemExit(f'Failed: see {log_path}')
    print('All reproduction steps completed:',out)


if __name__=='__main__':main()
