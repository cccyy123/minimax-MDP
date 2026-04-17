import math

def phase_reduction(diff_tau,matrix_L,L,matrix_R,R,V,delta,accuracy):
    """
    diff_tau: tau_{K}-tau_{K-1}
    matrix_L: n_L * K
    matrix_R: 1 * K
    L: n_L * n, L[i] denotes the value at i*accuracy
    R: 1 * n, R[i] denotes the value at i*accuracy
    V: 1*diff_tau
    delta: 1 * (1+diff_tau )
    """
    # defensive: if matrix_L is a 1D list, wrap it
    # Ensure matrix_L is 2D
    if len(matrix_L) > 0 and isinstance(matrix_L[0], (int, float)):
        matrix_L = [matrix_L]

# Ensure L is 2D (list of grids)
    if len(L) > 0 and isinstance(L[0], (int, float)):
        L = [L]
    n_L=len(matrix_L)
    K=len(matrix_R)
    n=len(R)
    inner_loop_iterations = 0
    new_matrix_R=matrix_R[:K-1]
    new_matrix_L=[]
    new_L=[]
    for coefficient, Lvalue in zip(matrix_L,L):
        #first kind
        templist=[]
        new_matrix_L.append(coefficient[:K-1])
        tempsum=coefficient[K-1]*sum(V)
        j_upper = math.floor((delta[0]-delta[diff_tau])/accuracy)+1
        for i in range(n):
            temp=-float('inf')
            for j in range(j_upper):
                inner_loop_iterations += 1
                temp=max( temp , Lvalue[min(n-1,i+j)] )
            templist.append(temp-tempsum)
        new_L.append(templist)

        #second kind
        if coefficient[K-1]!=0:
            templist=[]
            for i in range(K-1):
                templist.append(coefficient[i]/coefficient[K-1]-matrix_R[i]/matrix_R[K-1])
            new_matrix_L.append(templist)
            templist=[-float('inf')] * n
            for alpha in range(1,diff_tau+1):
                tempsum=sum(V[alpha:])
                for i in range(n):
                    temp=-float('inf')
                    for j in range(j_upper):
                        inner_loop_iterations += 1
                        temp=max(temp,  Lvalue[min(n-1,i+j)]/coefficient[K-1] - R[max( i , min(n-1,i+j-math.floor((delta[alpha-1]-delta[diff_tau])/accuracy) ) ) ]/matrix_R[K-1]   )
                    templist[i]=max(temp-tempsum,templist[i]) 
            new_L.append(templist)

    return new_matrix_L, new_L, new_matrix_R, inner_loop_iterations

