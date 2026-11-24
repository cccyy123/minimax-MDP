"""Optional NumPy evaluation of the archived DP recurrence (not for timing claims).

Preserves the original state grid, capped transitions, and last-action tie rule.
Original experiment solvers remain in constant_cost/ for literal replication.
"""
import math
import time
import numpy as np


def solve(T,L,R,V,demand,p,c,accuracy,opt,opt_count,ratio):
    start=time.process_time()
    N=math.ceil(sum(V)/accuracy)
    M=math.ceil(max(max(R),max(L),demand)/accuracy)
    lower=[math.floor(x/accuracy) for x in L]
    width=[math.ceil(r/accuracy)-l for r,l in zip(R,lower)]
    ii=np.arange(N+1)[:,None]
    jj=np.arange(M+1)[None,:]
    profit=p*np.minimum(ii,jj)-c*ii
    optimum=(p-c)*np.minimum(N,jj)
    if ratio:
        value=np.zeros((N+1,M+1))
        value[:,1:]=profit[:,1:]/optimum[:,1:]
    else:
        value=accuracy*(optimum-profit).astype(np.float64)
    actions=np.zeros((T,N+1,M+1),dtype=np.int32)
    for t in range(T-1,-1,-1):
        shift=max(width[t]-(width[t+1] if t+1<T else 0),0)
        # Original regret recurrence initializes the adversarial maximum at 0.
        env=np.full_like(value,np.inf) if ratio else np.zeros_like(value)
        for delta in range(shift+1):
            candidate=value[:,np.minimum(M,np.arange(M+1)+delta)]
            if ratio:np.minimum(env,candidate,out=env)
            else:np.maximum(env,candidate,out=env)
        best=np.full_like(value,-np.inf if ratio else np.inf)
        for order in range(math.floor(V[t]/accuracy)+1):
            candidate=env[np.minimum(N,np.arange(N+1)+order),:]
            take=candidate>=best if ratio else candidate<=best
            best[take]=candidate[take]
            actions[t][take]=order
        value=best
    inventory=0
    for t in range(T):
        inventory+=int(actions[t,inventory,min(M,max(0,lower[t]))])
    inventory*=accuracy
    reward=p*min(demand,inventory)-c*inventory
    return (reward/opt if ratio else opt-reward),time.process_time()-start+opt_count


def ratio_discrete_solver(*args):return solve(*args,ratio=True)
def regret_discrete_solver(*args):return solve(*args,ratio=False)
