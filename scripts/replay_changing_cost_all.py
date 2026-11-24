"""Recompute all 150 rounded-cost configurations x 100 archived seeds.

Uses original policy/search code and an exactly checked compiled reduction
kernel; writes fresh performance arrays and separate replay CPU measurements.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def initialize():
    sys.path.insert(0,str(ROOT/'changing_cost'))
    import rr,compiled_phase
    rr.phase_reduction.phase_reduction=compiled_phase.phase_reduction


def worker(job):
    import rr
    objective,group,seed,sample=job
    compute=rr._compute_ratio_single if objective=='ratio' else rr._compute_regret_single
    return group,seed,compute(sample,False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--groups',type=int,default=150)
    parser.add_argument('--seeds',type=int,default=100)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    os.environ['NUMBA_CACHE_DIR']=str(args.output/'numba-cache')
    summary=[]
    for objective in ['ratio','regret']:
        start=time.monotonic();folder=ROOT/'changing_cost/result'
        data=json.loads((folder/f'generate_decrease_K_multi_{objective}.json').read_text())
        reference=json.loads((folder/f'result_decrease_K_multi_{objective}.json').read_text())
        keys=['K_list','accuracy_func_list',f'accuracy_{objective}_list','tau_K_list','gamma','p',
              'L','R','T','totaldemand','V','true_K_list','true_tau_K_list','true_gamma_list']
        per_seed={'L','R','totaldemand','V'}
        def jobs():
            for group in range(args.groups):
                for seed in range(args.seeds):
                    sample=tuple(data[k][group][seed] if k in per_seed else data[k][group] for k in keys)
                    yield objective,group,seed,sample
        computed={f'performance_list_{m}':[[None]*args.seeds for _ in range(args.groups)] for m in ['up','lw','mid']}
        errors=[];largest=0;completed=0
        with (args.output/f'{objective}-samples.jsonl').open('w') as log:
            with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
                for group,seed,result in pool.map(worker,jobs(),chunksize=1):
                    worst=0
                    for method,index in [('up',1),('lw',3),('mid',5)]:
                        key=f'performance_list_{method}';value=float(result[index])
                        computed[key][group][seed]=value
                        error=abs(value-reference[key][group][seed]);worst=max(worst,error)
                        if error>1e-7:errors.append({'group':group,'seed':seed,'method':method,'actual':value,'reference':reference[key][group][seed],'error':error})
                    largest=max(largest,worst);completed+=1
                    log.write(json.dumps({'group':group,'seed':seed,'max_error':worst,
                                          'replay_cpu_times':[result[0],result[2],result[4]]})+'\n')
                    if completed%100==0:
                        log.flush();print(objective,completed,'/',args.groups*args.seeds,'max error',largest,flush=True)
                        (args.output/'progress.json').write_text(json.dumps({'objective':objective,'completed':completed,'total':args.groups*args.seeds,'max_error':largest}))
        (args.output/f'recomputed_{objective}.json').write_text(json.dumps(computed))
        record={'objective':objective,'configurations':args.groups,'seeds_per_configuration':args.seeds,
                'policy_values':3*completed,'max_error':largest,'mismatches':errors,'wall_seconds':time.monotonic()-start}
        summary.append(record)
        (args.output/'verification.json').write_text(json.dumps(summary,indent=2))
    if any(r['mismatches'] for r in summary):raise SystemExit('Mismatch: see verification.json')


if __name__=='__main__':main()
