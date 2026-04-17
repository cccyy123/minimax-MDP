import json,math
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ins_acc_list=[1.02,1.05,1.08]
num_ins_acc=len(ins_acc_list)
w=2
true_tau_K_list=[[0,65,75,115,155,199] for _ in range(w)]
true_K_list=[len(x)-1 for x in true_tau_K_list]
true_gamma_list=[[4.7,4.8,4.9,5.0,5.1] for _ in range(w)]
p_list=[10] * w

def accuracy_return_K_gamma(true_tau_K, true_gamma, p, delta=2):
    if delta <= 1:
        raise ValueError("delta must be > 1")
    if len(true_gamma) == 0:
        raise ValueError("true_gamma must be non-empty")
    if any(true_gamma[i] < true_gamma[i - 1] for i in range(1, len(true_gamma))):
        raise ValueError("true_gamma must be nondecreasing")

    temp1=[]
    temp1.append(true_gamma[0])
    while temp1[-1]<=true_gamma[-1]:
        temp1.append(temp1[-1]*delta)
    temp2=[]
    temp2.append(true_gamma[-1])
    while (p-temp2[-1])<=p-true_gamma[0]:
        temp2.append(p-(p-temp2[-1])*delta)
    temp2.reverse()
    candidate_gamma=temp1[:]+temp2[:]
    candidate_gamma=sorted(list(set(candidate_gamma)))
    k=0
    gamma=[x for x in true_gamma]
    for i in range(len(gamma)):
        while candidate_gamma[k+1]<=gamma[i]:
            k+=1
        gamma[i]=candidate_gamma[k]
    return_tau_K=[0]
    return_gamma=[gamma[0]]
    for i in range(1,len(gamma)):
        if gamma[i]>gamma[i-1]:
            return_tau_K.append(true_tau_K[i])
            return_gamma.append(gamma[i])
    return_tau_K.append(true_tau_K[-1])
    return return_tau_K, return_gamma



for ins_acc in ins_acc_list:
    tau_K, gamma=accuracy_return_K_gamma(true_tau_K_list[0], true_gamma_list[0], p_list[0], delta=ins_acc)
    print(tau_K, gamma)