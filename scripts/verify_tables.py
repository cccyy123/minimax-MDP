"""Re-evaluate archived constant-cost inputs without regenerating random data."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import copy
import statistics
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]


def evaluate_sample(job):
    script, data, dp, vectorized = job
    spec = importlib.util.spec_from_file_location('table_solver', ROOT/'constant_cost'/script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if vectorized:
        import vectorized_dp
        module.ratio_discrete_solver=vectorized_dp.ratio_discrete_solver
        module.regret_discrete_solver=vectorized_dp.regret_discrete_solver
    return module.run_ratio_and_regret(data, run_discrete=dp)


def evaluate_parallel(script, data, dp, workers, vectorized=False):
    # Archived table datasets have one parameter group and independent seeds.
    if len(data['T']) != 1:
        raise ValueError('Parallel table verification expects one parameter group')
    jobs=[]
    for seed in range(data['seed_repeat']):
        single=copy.deepcopy(data)
        for key in ['V','L','R','Delta','totaldemand']:
            single[key]=[[data[key][0][seed]]]
        single['seed_repeat']=1
        jobs.append((script,single,dp,vectorized))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        samples=[]
        for index,value in enumerate(pool.map(evaluate_sample,jobs),1):
            samples.append(value)
            if index%10==0: print(script, index, '/', len(jobs), flush=True)
    combined=copy.deepcopy(samples[0])
    for objective in [0,1]:
        combined[objective]['seed_repeat']=len(samples)
        for method,rows in combined[objective]['methods'].items():
            for key in rows[0]:
                mean_key=key[:-4]+'_mean' if key.endswith('_std') else key
                values=[sample[objective]['methods'][method][0][mean_key] for sample in samples]
                rows[0][key]=statistics.pstdev(values) if key.endswith('_std') else statistics.mean(values)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dp', action='store_true', help='Also rerun both DP baselines (expensive).')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--vectorized-dp',action='store_true',help='Evaluate the same DP recurrence with NumPy; not comparable timing.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = []
    for suffix in ['', '_perfect_only']:
        script = 'generating-table-no-shift-execute' + ('-perfect-only' if suffix else '') + '.py'
        spec = importlib.util.spec_from_file_location('table_solver', ROOT/'constant_cost'/script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if args.vectorized_dp:
            import vectorized_dp
            module.ratio_discrete_solver=vectorized_dp.ratio_discrete_solver
            module.regret_discrete_solver=vectorized_dp.regret_discrete_solver
        data = json.loads((ROOT/f'constant_cost/results/generate_table{suffix}.json').read_text())
        start = time.monotonic()
        if args.workers > 1:
            ratio, regret = evaluate_parallel(script, data, args.dp, args.workers,args.vectorized_dp)
        else:
            ratio, regret = module.run_ratio_and_regret(data, run_discrete=args.dp)
        errors = []
        for objective, result in [('ratio',ratio),('rgt',regret)]:
            (args.output/f'result_{objective}{suffix}.json').write_text(json.dumps(result,indent=2))
            reference = json.loads((ROOT/f'constant_cost/results/result_{objective}_table{suffix}.json').read_text())
            for method, rows in result['methods'].items():
                if not args.dp and method.startswith('dis_'): continue
                for idx, row in enumerate(rows):
                    for key, value in row.items():
                        if key.startswith('time'): continue
                        error = abs(value-reference['methods'][method][idx][key])
                        errors.append({'objective':objective,'method':method,'statistic':key,'error':error})
        report.append({'dataset':suffix or 'noisy','runs':data['seed_repeat'],
                       'dp_included':args.dp,'vectorized_dp':args.vectorized_dp,'wall_seconds':time.monotonic()-start,
                       'max_absolute_error':max(x['error'] for x in errors),'comparisons':errors})
        print(report[-1]['dataset'], 'max error', report[-1]['max_absolute_error'], flush=True)
    (args.output/'verification.json').write_text(json.dumps(report,indent=2))
    if any(r['max_absolute_error']>1e-7 for r in report):
        raise SystemExit('Archived result mismatch; inspect verification.json')


if __name__ == '__main__': main()
