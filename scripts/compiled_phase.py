"""Optional compilation of the original phase-reduction loops, without fastmath.

Only use for numerical replay, not reproduction of historical CPU timings.
"""
import math
import numpy as np
from numba import njit
from numerics import sequential_sum as sum


@njit(cache=True, fastmath=False)
def kernel(diff_tau, matrix_L, L, matrix_R, R, capacity_sums, delta, accuracy):
    K=len(matrix_R);n=len(R);rows=len(matrix_L)
    size=rows
    for row in range(rows):
        if matrix_L[row,K-1]!=0:size+=1
    new_A=np.empty((size,K-1));new_L=np.empty((size,n))
    out=0;count=0
    for row in range(rows):
        coefficient=matrix_L[row];values=L[row]
        for k in range(K-1):new_A[out,k]=coefficient[k]
        total=coefficient[K-1]*capacity_sums[0]
        j_upper=math.floor((delta[0]-delta[diff_tau])/accuracy)+1
        for i in range(n):
            best=-math.inf
            for j in range(j_upper):
                count+=1
                best=max(best,values[min(n-1,i+j)])
            new_L[out,i]=best-total
        out+=1
        if coefficient[K-1]!=0:
            for k in range(K-1):
                new_A[out,k]=coefficient[k]/coefficient[K-1]-matrix_R[k]/matrix_R[K-1]
            for i in range(n):new_L[out,i]=-math.inf
            for alpha in range(1,diff_tau+1):
                total=capacity_sums[alpha]
                for i in range(n):
                    best=-math.inf
                    for j in range(j_upper):
                        count+=1
                        ri=max(i,min(n-1,i+j-math.floor((delta[alpha-1]-delta[diff_tau])/accuracy)))
                        candidate=values[min(n-1,i+j)]/coefficient[K-1]-R[ri]/matrix_R[K-1]
                        best=max(best,candidate)
                    new_L[out,i]=max(best-total,new_L[out,i])
            out+=1
    return new_A,new_L,matrix_R[:K-1],count


def phase_reduction(diff_tau,matrix_L,L,matrix_R,R,V,delta,accuracy):
    A=np.atleast_2d(np.asarray(matrix_L,dtype=np.float64))
    values=np.atleast_2d(np.asarray(L,dtype=np.float64))
    a,b,c,count=kernel(diff_tau,A,values,np.asarray(matrix_R,dtype=np.float64),
                       np.asarray(R,dtype=np.float64),np.asarray([sum(V[a:]) for a in range(diff_tau+1)]),
                       np.asarray(delta,dtype=np.float64),accuracy)
    return a.tolist(),b.tolist(),c.tolist(),count
