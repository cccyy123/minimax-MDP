"""Check selected archived seeds/configurations with the original solver."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'changing_cost'))
import rr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+', type=int, default=[10,60,110])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report=[]
    for objective in ['ratio','regret']:
        folder=ROOT/'changing_cost/result'
        data=json.loads((folder/f'generate_decrease_K_multi_{objective}.json').read_text())
        ref=json.loads((folder/f'result_decrease_K_multi_{objective}.json').read_text())
        keys=['K_list','accuracy_func_list',f'accuracy_{objective}_list','tau_K_list','gamma','p',
              'L','R','T','totaldemand','V','true_K_list','true_tau_K_list','true_gamma_list']
        per_seed={'L','R','totaldemand','V'}
        for group in args.groups:
            for seed in args.seeds:
                sample=tuple(data[k][group][seed] if k in per_seed else data[k][group] for k in keys)
                compute=rr._compute_ratio_single if objective=='ratio' else rr._compute_regret_single
                result=compute(sample, False)
                errors={m:abs(result[i]-ref[f'performance_list_{m}'][group][seed])
                        for m,i in [('up',1),('lw',3),('mid',5)]}
                record={'objective':objective,'group':group,'seed_index':seed,
                        'max_absolute_error':max(errors.values()),'errors':errors}
                report.append(record)
                print(record,flush=True)
                (args.output/'verification.json').write_text(json.dumps(report,indent=2))
    if any(r['max_absolute_error']>1e-7 for r in report):
        raise SystemExit('Mismatch with archived performance values')


if __name__=='__main__': main()
