import json,math
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ins_acc_list=[1.02,1.05,1.08]
num_ins_acc=len(ins_acc_list)
num_sol_acc=50  #50
w=(num_ins_acc+1)*num_sol_acc
n_mc=400  #500
num_seeds_per_example = 100 #100
# 最多使用的 CPU 核数；设为 0 或负数表示使用全部可用核心
max_cores = 100

T_list = [200 ]*w

true_tau_K_list=[[0,65,75,115,155,199] for _ in range(w)]
true_K_list=[len(x)-1 for x in true_tau_K_list]
true_gamma_list=[[4.7,4.8,4.9,5.0,5.1] for _ in range(w)]
p_list=[10] * w

accuracy_regret_list=[2] * w
accuracy_ratio_list=[0.01] * w
maxpi_list = [0.5] * w
rangedemand_list = [5] * w
V_p_list = [0.5]* w
shared_seed_list = [5769 + i for i in range(num_seeds_per_example)]
n_mc_list = [ n_mc ] * w
lower_q_list = [0.05] * w
upper_q_list = [0.95] * w
V_list=[  ]
L_list=[  ]
R_list=[  ]
Delta_list=[]
demand_list=[]
seed_list=[]


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



accuracy_func_list=[ 1+ 25*i/(num_sol_acc-1) for i in range(num_sol_acc)]*(num_ins_acc+1)  #for discrete
tau_K_list=[]
gamma_list=[]
for ins_acc in ins_acc_list:
    for _ in range(num_sol_acc):
        tau_K, gamma=accuracy_return_K_gamma(true_tau_K_list[0], true_gamma_list[0], p_list[0], delta=ins_acc)
        tau_K_list.append(tau_K)
        gamma_list.append(gamma)
for _ in range(num_sol_acc):
    tau_K_list.append(true_tau_K_list[0])
    gamma_list.append(true_gamma_list[0])
K_list=[len(x)-1 for x in tau_K_list]


def calibrate_asymmetric_intervals(
    pi,
    estdemand,
    rangedemand,
    n_mc=20000,
    lower_q=0.025,
    upper_q=0.975,
    seed=None,
):
    """
    离线校准：对每个 i 构造不对称误差区间 [q_lower(i), q_upper(i)]
    使得 P(q_lower(i) <= e_i <= q_upper(i)) ≈ (upper_q - lower_q)

    误差定义（与你原代码一致）：
    e_i = true_future(i) - sample_mean_future(i)

    在线区间（对“true_future”的预测）应当用：
    L_i = estdemand[i] + q_lower(i)
    R_i = estdemand[i] + q_upper(i)

    参数：
    pi: 长度 T 的列表/数组，每天的成功概率
    estdemand: 长度 T 的点预测（和 e_i 对应的目标量的预测；通常应是“未来总和”的点预测）
    rangedemand: 二项分布 n（你的 rangedemand）
    n_mc: Monte Carlo 次数
    lower_q / upper_q: 分位点（可调控），例如 0.05 和 0.95
    seed: 随机种子（可选）

    返回：
    q_lower_list, q_upper_list, L, R, coverage
        coverage = upper_q - lower_q
    """
    if seed is not None:
        np.random.seed(seed)

    pi = np.asarray(pi, dtype=float)
    estdemand = np.asarray(estdemand, dtype=float)
    T = len(pi)

    if len(estdemand) != T:
        raise ValueError(f"len(estdemand)={len(estdemand)} must equal len(pi)={T}")
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError("Require 0 <= lower_q < upper_q <= 1")

    # 存每个 i 的误差样本 e_i（不取绝对值）
    err = [[] for _ in range(T)]

    for _ in range(n_mc):
        # (a) 固定 pi 下抽真实 phi_mc
        phi_mc = np.random.binomial(rangedemand, pi, size=T)

        # (b) 固定 pi 下生成每天的 sampled future trajectory estphi_mc
        # estphi_mc[t, k] only meaningful for k > t
        estphi_mc = np.zeros((T, T), dtype=float)
        for t in range(T):
            if t + 1 < T:
                estphi_mc[t, t+1:] = np.random.binomial(
                    rangedemand, pi[t+1:], size=T-(t+1)
                )

        # (c) 对每一天 i 计算 e_i = true_future - (average sampled future)
        for i in range(T):
            true_future = float(phi_mc[i+1:].sum())

            # sample_mean_future = (1/(i+1)) * sum_{tau<=i} sum_{k>i} estphi_mc[tau,k]
            if i + 1 < T:
                sample_sum = float(estphi_mc[:i+1, i+1:].sum())
            else:
                sample_sum = 0.0

            sample_mean_future = sample_sum / (i + 1)
            e = true_future - sample_mean_future
            err[i].append(e)

    # (d) 分别取左右分位数 -> 不对称半径（其实就是不对称误差边界）
    q_lower_list = [float(np.quantile(err[i], lower_q)) for i in range(T)]
    q_upper_list = [float(np.quantile(err[i], upper_q)) for i in range(T)]

    # ===== 在线区间：注意这里是 +（因为 true = est + e）=====
    L = (estdemand + np.asarray(q_lower_list)).tolist()
    R = (estdemand + np.asarray(q_upper_list)).tolist()

    coverage = upper_q - lower_q
    return q_lower_list, q_upper_list, L, R, coverage


def run_one_seed(args):
    seed, T, maxpi, rangedemand, V_p, n_mc, lower_q, upper_q = args

    # ===== 1) 先生成整条 pi，再生成一次真实 phi（你原本的逻辑）=====
    pi = [0] * T
    phi = [0] * T

    np.random.seed(seed)
    for i in range(T):
        pi[i] = np.random.uniform(0, maxpi)
        phi[i] = np.random.binomial(rangedemand, pi[i])

    # ===== 2) 计算一次在线估计 estdemand（你原本的逻辑）=====
    estphi = [[0] * T for _ in range(T)]
    estdemand = [0] * T

    currenttotaldemand = 0
    currentfuturedemand = 0
    esteachdemand = [0] * T

    for i in range(T):
        currenttotaldemand += phi[i]
        for k in range(i + 1, T):
            estphi[i][k] = np.random.binomial(rangedemand, pi[k])
            currentfuturedemand += estphi[i][k]
            esteachdemand[k] += estphi[i][k]
        currentfuturedemand -= esteachdemand[i]
        estdemand[i] = currenttotaldemand + currentfuturedemand / (i + 1)

    qL, qU, L, R, cov = calibrate_asymmetric_intervals(
        pi=pi,
        estdemand=estdemand,
        rangedemand=rangedemand,
        n_mc=n_mc,
        lower_q=lower_q,
        upper_q=upper_q,
        seed=seed,
    )
    Delta = [r - l for r, l in zip(R, L)]

    V = [sum(phi) * 3 * t / (T * (T - 1)) for t in range(T - 1, 0, -1)]
    # rng = np.random.default_rng(seed=seed)
    # V = rng.binomial(math.ceil(sum(phi)/((T-1)*V_p)), V_p, size=T-1).tolist()

    return {
        "seed": seed,
        "V": V,
        "Delta": Delta,
        "L": L,
        "R": R,
        "demand": sum(phi),
    }


def main():
    max_available_workers = os.cpu_count() or 1
    if max_cores > 0:
        num_workers = min(max_cores, max_available_workers)
    else:
        num_workers = max_available_workers

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for T, maxpi, rangedemand, V_p, n_mc, lower_q, upper_q in zip(
            T_list,
            maxpi_list,
            rangedemand_list,
            V_p_list,
            n_mc_list,
            lower_q_list,
            upper_q_list,
        ):
            args_list = [
                (seed, T, maxpi, rangedemand, V_p, n_mc, lower_q, upper_q)
                for seed in shared_seed_list
            ]
            seed_results = list(executor.map(run_one_seed, args_list))

            V_runs = [item["V"] for item in seed_results]
            Delta_runs = [item["Delta"] for item in seed_results]
            L_runs = [item["L"] for item in seed_results]
            R_runs = [item["R"] for item in seed_results]
            demand_runs = [item["demand"] for item in seed_results]
            current_seed_runs = [item["seed"] for item in seed_results]

            V_list.append(V_runs)
            Delta_list.append(Delta_runs)
            L_list.append(L_runs)
            R_list.append(R_runs)
            demand_list.append(demand_runs)
            seed_list.append(current_seed_runs)

    data_regret = {'K_list':K_list,'accuracy_func_list':accuracy_func_list,'accuracy_regret_list':accuracy_regret_list,'tau_K_list':tau_K_list,"V":V_list, "gamma":gamma_list, "p":p_list, 'L': L_list, 'R': R_list, 'Delta':Delta_list, 'T':T_list, 'totaldemand':demand_list,'seed_list':seed_list, 'true_K_list': true_K_list, 'true_tau_K_list': true_tau_K_list, 'true_gamma_list': true_gamma_list}
    data_ratio = {'K_list':K_list,'accuracy_func_list':accuracy_func_list,'accuracy_ratio_list':accuracy_ratio_list,'tau_K_list':tau_K_list,"V":V_list, "gamma":gamma_list, "p":p_list, 'L': L_list, 'R': R_list, 'Delta':Delta_list, 'T':T_list, 'totaldemand':demand_list,'seed_list':seed_list, 'true_K_list': true_K_list, 'true_tau_K_list': true_tau_K_list, 'true_gamma_list': true_gamma_list}

    results_dir = Path(__file__).resolve().parent / "result"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "generate_decrease_K_multi_regret.json", "w", encoding="utf-8") as f:
        json.dump(data_regret, f, ensure_ascii=False, indent=4)

    with open(results_dir / "generate_decrease_K_multi_ratio.json", "w", encoding="utf-8") as f:
        json.dump(data_ratio, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()

