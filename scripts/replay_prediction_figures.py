"""Recompute all displayed shift/demand-horizon policy values from input samples."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def evaluate(L,R,V,demand,p,c,ratio):
    T=len(V);L=L[:T];R=R[:T];delta=[r-l for l,r in zip(L,R)]
    suffix=[sum(V[t+1:T]) for t in range(T)]
    opt=(p-c)*min(demand,sum(V))
    if ratio:
        value=min((p*L[0]+c*suffix[t])/(p*L[0]+c*delta[t]) for t in range(T))
        lo=[R[t]*value-suffix[t] for t in range(T)]
        hi=[L[t]*((1-value)*p+value*c)/c for t in range(T)]
    else:
        value=c*(p-c)/p*max(delta[t]-suffix[t] for t in range(T))
        lo=[R[t]-value/(p-c)-suffix[t] for t in range(T)]
        hi=[L[t]+value/c for t in range(T)]
    results={}
    for method in ['up','lw','mid']:
        inventory=0
        for t in range(T):
            target=hi[t] if method=='up' else lo[t] if method=='lw' else (hi[t]+lo[t])/2
            inventory+=min(max(0,target-inventory),V[t])
        reward=p*min(demand,inventory)-c*inventory
        results['rgt_'+method]=reward/opt if ratio else opt-reward
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for family in ['shift','demand_horizon']:
        folder=ROOT/'constant_cost/results'
        data=json.loads((folder/f'generate_{family}.json').read_text())
        for obj in ['ratio','rgt']:
            ref=json.loads((folder/f'result_{obj}_{family}.json').read_text())
            computed={k:[] for k in ['rgt_up','rgt_lw','rgt_mid']}
            largest=0;count=0
            for group in range(len(data['T'])):
                rows={k:[] for k in computed}
                for seed in range(len(data['L'][group])):
                    vals=evaluate(data['L'][group][seed],data['R'][group][seed],data['V'][group][seed],
                                  data['totaldemand'][group][seed],data['p'][group],data['c'][group],obj=='ratio')
                    for key,value in vals.items():
                        rows[key].append(value);count+=1
                        largest=max(largest,abs(value-ref[key][group][seed]))
                for key in computed:computed[key].append(rows[key])
            (args.output/f'result_{obj}_{family}.json').write_text(json.dumps(computed))
            record={'family':family,'objective':obj,'policy_values':count,'max_error':largest}
            records.append(record);print(record,flush=True)
    (args.output/'verification.json').write_text(json.dumps(records,indent=2))
    if any(r['max_error']>1e-7 for r in records):raise SystemExit('Archived result mismatch')


if __name__=='__main__':main()
