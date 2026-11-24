"""Build paper plots with replayed performance and archived timing measurements."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--changing-cost',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out==ROOT or ROOT in out.parents or out in ROOT.parents:
        parser.error('Choose a fresh output directory outside the repository')
    out.mkdir(parents=True,exist_ok=False)
    for folder in ['constant_cost','changing_cost']:
        shutil.copytree(ROOT/folder,out/folder,ignore=shutil.ignore_patterns('__pycache__'))
    for family in ['shift','demand_horizon']:
        for objective in ['ratio','rgt']:
            name=f'result_{objective}_{family}.json'
            shutil.copyfile(args.predictions/name,out/'constant_cost/results'/name)
    for objective in ['ratio','regret']:
        dest=out/f'changing_cost/result/result_decrease_K_multi_{objective}.json'
        data=json.loads(dest.read_text())
        fresh=json.loads((args.changing_cost/f'recomputed_{objective}.json').read_text())
        for key,values in fresh.items():data[key][:len(values)]=values
        dest.write_text(json.dumps(data))
    env=dict(os.environ,MPLBACKEND='Agg',MPLCONFIGDIR=str(out/'.matplotlib'))
    for script in ['constant_cost/results/plot_temp_horizon.py','constant_cost/results/plot_shift.py',
                   'constant_cost/results/plot_demand_horizon.py','changing_cost/plot_perf_time_scatter.py']:
        path=out/script
        with (out/(path.stem+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(path)],cwd=path.parent,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'PROVENANCE.json').write_text(json.dumps({
        'performance':'Freshly recomputed shift/demand-horizon and 150 changing-cost configurations',
        'timings':'Historical archived measurements, not accelerated replay timings',
        'plot_selection':'All 150 rounded-cost configurations; final 50 exact-cost reference configurations excluded',
    },indent=2))
    print('Plots regenerated at',out)


if __name__=='__main__':main()
