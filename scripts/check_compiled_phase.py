"""Check compiled phase reduction against original loops before full replay."""
from pathlib import Path
import sys
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'changing_cost'))
import phase_reduction
import compiled_phase

rng=np.random.default_rng(923)
records=[]
for index in range(60):
    k=index%5+1;rows=index%3+1;n=index%9+2;dt=index%7+1
    A=rng.uniform(.1,3,(rows,k)).tolist()
    if index%4==0:A[0][-1]=0
    L=rng.uniform(-5,10,(rows,n)).tolist();B=rng.uniform(.2,3,k).tolist()
    R=rng.uniform(-2,8,n).tolist();V=rng.uniform(0,5,dt).tolist()
    delta=sorted(rng.uniform(0,5,dt+1).tolist(),reverse=True)
    args=(dt,A,L,B,R,V,delta,.5)
    a=phase_reduction.phase_reduction(*args);b=compiled_phase.phase_reduction(*args)
    assert a==b,(index,a,b)
    records.append({'case':index,'exact_match':True,'inner_iterations':a[3]})
print(json.dumps({'seed':923,'cases':records},indent=2))
