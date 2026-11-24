"""Export complete archived constant-cost statistics and EC.7 phase counts."""
import argparse
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for suffix in ['', '_perfect_only']:
        for objective in ['ratio','rgt']:
            data=json.loads((ROOT/f'constant_cost/results/result_{objective}_table{suffix}.json').read_text())
            for method,groups in data['methods'].items():
                for index,values in enumerate(groups):
                    for statistic,value in values.items():
                        rows.append([suffix or 'noisy',objective,method,index,data['seed_repeat'],statistic,value])
    with (args.output/'constant_cost_tables.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['dataset','objective','method','group','runs','statistic','value']);writer.writerows(rows)
    data=json.loads((ROOT/'changing_cost/result/generate_decrease_K_multi_regret.json').read_text())
    result=json.loads((ROOT/'changing_cost/result/result_decrease_K_multi_regret.json').read_text())
    with (args.output/'phase_counts_and_times.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['delta_ins','demand_grid','phases','constraints','mean_archived_seconds'])
        for group,delta in [(10,.02),(60,.05),(110,.08)]:
            k=data['K_list'][group]
            writer.writerow([delta,data['accuracy_func_list'][group],k,2**k+1,statistics.mean(result['time_list_up'][group])])
    print('Exported tables to',args.output)


if __name__=='__main__': main()
