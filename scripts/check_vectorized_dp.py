"""Cross-check NumPy recurrence against the unchanged original DP on small cases."""
import importlib.util
import json
from pathlib import Path
import random
import vectorized_dp

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('original',ROOT/'constant_cost/generating-table-no-shift-execute.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
rng=random.Random(923)
checks=[]
for index in range(40):
    T=rng.randint(1,5);p=rng.randint(2,5);c=1
    V=[rng.choice([.5,1,2,3]) for _ in range(T)]
    L=[rng.choice([0,.5,1,2]) for _ in range(T)]
    R=[l+rng.choice([0,.5,1,2,3]) for l in L]
    demand=rng.choice([1,2,3]);accuracy=rng.choice([.5,1,2])
    opt=(p-c)*min(demand,sum(V));args=(T,L,R,V,demand,p,c,accuracy,opt,0)
    for name in ['ratio_discrete_solver','regret_discrete_solver']:
        a=getattr(original,name)(*args)[0];b=getattr(vectorized_dp,name)(*args)[0]
        checks.append({'case':index,'solver':name,'original':a,'vectorized':b,'error':abs(a-b)})
        assert abs(a-b)<1e-12,(index,name,a,b)
print(json.dumps({'seed':923,'checks':checks,'max_error':max(x['error'] for x in checks)},indent=2))
