import json,math
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


w=101
seed_repeat = 80
base_seed = 10
seed_values = [base_seed + i for i in range(seed_repeat)]

T_list = [200 ]*w
maxpi_list = [0.5] * w
rangedemand_list = [5] * w
c_list=[1]* w
p_list=[2] * w
shift_list=[-25+50*i/(w-1) for i in range(w) ]
#shift_list=[0.1*((1.14)**i) for i in range(rr)]+[(-0.1*((1.14)**i)) for i in range(rr)]  # 0.1*(1.14**i) for i in range(w)
shift_prob_list=[0.3 for _ in range(w)]
n_mc_list = [ 100 ] * w
V_list=[  ]
L_list=[  ]
R_list=[  ]
Delta_list=[]
demand_list=[]
shift_indices_list=[]

for T,maxpi,rangedemand,c,p,n_mc,shift,shift_prob in zip(
    T_list,
    maxpi_list,
    rangedemand_list,
    c_list,
    p_list,
    n_mc_list,
    shift_list,
    shift_prob_list,
):
    group_V = []
    group_Delta = []
    group_L = []
    group_R = []
    group_demand = []
    group_shift_indices = []

    # ===== 3) 离线校准：对称区间半径 l_i，使 P(|e_i|<=l_i)=95% =====
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


    for seed in seed_values:
        # ===== 1) 先生成整条 pi，再生成一次真实 phi（你原本的逻辑）=====
        pi = [0]*T
        phi = [0]*T

        np.random.seed(seed)
        for i in range(T):
            pi[i] = np.random.uniform(0, maxpi)
            phi[i] = np.random.binomial(rangedemand, pi[i])

        # ===== 2) 计算一次在线估计 estdemand（你原本的逻辑）=====
        estphi = [[0]*T for _ in range(T)]
        estdemand = [0] * T

        currenttotaldemand  = 0
        currentfuturedemand = 0
        esteachdemand = [0] * T

        for i in range(T):
            currenttotaldemand += phi[i]
            for k in range(i+1, T):
                estphi[i][k] = np.random.binomial(rangedemand, pi[k])
                currentfuturedemand += estphi[i][k]
                esteachdemand[k] += estphi[i][k]
            currentfuturedemand -= esteachdemand[i]
            estdemand[i] = currenttotaldemand + currentfuturedemand/(i+1)

        qL, qU, L, R, cov = calibrate_asymmetric_intervals(
            pi=pi,
            estdemand=estdemand,
            rangedemand=rangedemand,
            n_mc=n_mc,
            lower_q=0.05,
            upper_q=0.95,
            seed=seed
        )

        # IID per-index shift: each index is shifted with probability shift_prob.
        if not (0.0 <= shift_prob <= 1.0):
            raise ValueError(f"shift_prob must be in [0,1], got {shift_prob}")

        rng_shift = np.random.default_rng(seed)
        shift_mask = rng_shift.random(T) < shift_prob
        shifted_indices = np.where(shift_mask)[0].tolist()
        for j in shifted_indices:
            L[j] = max(0.0, L[j] + shift)
            R[j] = max(0.0, R[j] + shift)

        Delta=[r-l for r,l in zip(R,L)]

        V=[ sum(phi)*3*t/(T*(T-1)) for t in range(T-1,0,-1)]
        #V=np.random.binomial(math.ceil(sum(phi)/((T-1)*V_p)), V_p, size=T-1).tolist()
        group_V.append(V)
        group_Delta.append(Delta)
        group_L.append(L)
        group_R.append(R)
        group_demand.append(sum(phi))
        group_shift_indices.append(shifted_indices)

    V_list.append(group_V)
    Delta_list.append(group_Delta)
    L_list.append(group_L)
    R_list.append(group_R)
    demand_list.append(group_demand)
    shift_indices_list.append(group_shift_indices)

data = {
    'V': V_list,
    'c': c_list,
    'p': p_list,
    'L': L_list,
    'R': R_list,
    'Delta': Delta_list,
    'T': T_list,
    'totaldemand': demand_list,
    'shift_indices': shift_indices_list,
    'shift_prob': shift_prob_list,
    'shift_value': shift_list,
    'seed_repeat': seed_repeat,
    'seed_values': seed_values,
}

with open("results/generate_shift.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False,indent=4)




import json
import time,math

# Global switch: whether to run dis_1 and dis_2 discrete solvers.
RUN_DISCRETE_TEST = False

with open("results/generate_shift.json", "r", encoding="utf-8") as f:
    data = json.load(f)


accuracy_1=1
accuracy_2=4

def to_repeat_list(group_value, expected_repeat):
    if expected_repeat <= 0:
        raise ValueError(f"seed_repeat must be positive, got {expected_repeat}")

    if isinstance(group_value, list):
        # New format: per-parameter group already expanded by seed_repeat.
        if len(group_value) == expected_repeat:
            return group_value
        # Backward compatibility: single-run legacy shape.
        if expected_repeat == 1:
            return [group_value]
    elif expected_repeat == 1:
        return [group_value]

    raise ValueError(
        f"Input data does not match seed_repeat shape: "
        f"type={type(group_value).__name__}, "
        f"len={len(group_value) if isinstance(group_value, list) else 'NA'}, "
        f"seed_repeat={expected_repeat}"
    )


seed_repeat = data.get("seed_repeat", 1)

opt_total=[]
r_lw_total=[]
r_mid_total=[]
r_up_total=[]
r_dis_1_total=[]
r_dis_2_total=[]
lw_time=[]
up_time=[]
mid_time=[]
dis_1_time=[]
dis_2_time=[]

for T,L_group,R_group,delta_group,demand_group,V_group,c,p in zip(
    data['T'],
    data['L'],
    data['R'],
    data['Delta'],
    data['totaldemand'],
    data['V'],
    data['c'],
    data['p']
):
    T=T-1

    L_repeats = to_repeat_list(L_group, seed_repeat)
    R_repeats = to_repeat_list(R_group, seed_repeat)
    delta_repeats = to_repeat_list(delta_group, seed_repeat)
    V_repeats = to_repeat_list(V_group, seed_repeat)
    demand_repeats = to_repeat_list(demand_group, seed_repeat)

    group_opt=[]
    group_r_lw=[]
    group_r_mid=[]
    group_r_up=[]
    group_r_dis_1=[]
    group_r_dis_2=[]
    group_lw_time=[]
    group_up_time=[]
    group_mid_time=[]
    group_dis_1_time=[]
    group_dis_2_time=[]

    for L,R,delta,demand,V in zip(L_repeats, R_repeats, delta_repeats, demand_repeats, V_repeats):
        L=L[:T]
        R=R[:T]
        delta=delta[:T]
        opt = (p-c) * min(demand, sum(V))

        # our method: including two extreme greedy and their mid
        start_opt= time.perf_counter()
        opt_gamma=min( [ ( p*L[0] + c*sum(V[i+1:T]) )/( p*L[0] + c*delta[i]  ) for i in range(T)])
        opt_count=time.perf_counter()-start_opt
        group_opt.append(opt_gamma)

        inv_mid=0
        inv_lw=0
        inv_up=0

        #mid
        start_our_mid=time.perf_counter()
        for t in range(T):
            a_mid=( R[t]*opt_gamma-  sum(V[t+1:T]) +  L[t] * ( (1-opt_gamma)*p +opt_gamma*c )  /c   )/2-inv_mid
            a_mid= min(max(0,a_mid),V[t])
            inv_mid+=a_mid

        alg_mid = p*min(demand,inv_mid)-c*inv_mid
        ratio_mid=alg_mid/opt
        our_mid_count=time.perf_counter()-start_our_mid+opt_count

        group_mid_time.append(our_mid_count)
        group_r_mid.append(ratio_mid)

        #lw
        start_our_lw=time.perf_counter()
        for t in range(T):
            a_lw=R[t]*opt_gamma-  sum(V[t+1:T])-inv_lw
            a_lw=min(max(0,a_lw),V[t])
            inv_lw += a_lw

        alg_lw = p*min(demand,inv_lw)-c*inv_lw
        ratio_lw=alg_lw/opt
        our_lw_count=time.perf_counter()-start_our_lw+opt_count

        group_lw_time.append(our_lw_count)
        group_r_lw.append(ratio_lw)


        #up
        start_our_up=time.perf_counter()
        for t in range(T):
            a_up=  L[t] * ( (1-opt_gamma)*p +opt_gamma*c )  /c  - inv_up
            a_up=min(max(0,a_up),V[t])
            inv_up+=a_up

        alg_up = p*min(demand,inv_up)-c*inv_up
        ratio_up = alg_up/opt
        our_up_count=time.perf_counter()-start_our_up+opt_count

        group_up_time.append(our_up_count)
        group_r_up.append(ratio_up)


        if RUN_DISCRETE_TEST:
            # discrete optimization 1,2,3,...,T+1 state[T][N][M]=[rgt,a]
            N=math.ceil(sum(V)/accuracy_1) #inv 0,1,...,N
            M=math.ceil(max(max(R),max(L),demand)/accuracy_1) #left 0,1,...,M
            discrete_L=[math.floor(a/accuracy_1) for a in L]
            discrete_R=[math.ceil(a/accuracy_1) for a in R]
            discrete_delta=[x-y for x,y in zip(discrete_R,discrete_L)]
            state=[  [  [  [0,0] for _ in range(M+1)]    for _ in range(N+1)]   for _ in range(T+1)   ]
            state_env=[  [  [  0 for _ in range(M+1)]    for _ in range(N+1)]   for _ in range(T) ]

            start_our_dis=time.perf_counter()
            for i in range(N+1):
                for j in range(1,M+1):
                    state[T][i][j]=[  (p*min(i,j)-c*i) / ( (p-c)*min(N,j) ),0]

            for t in range(T-1,-1,-1):
                for i in range(N+1):
                    for j in range(M+1):
                        tempmin=float('inf')
                        next_delta = discrete_delta[t+1] if t+1 < T else 0
                        for add_j in range(max(discrete_delta[t]-next_delta,0)+1):
                            tempmin=min(tempmin,state[t+1][i][min(M,j+add_j)][0])
                        state_env[t][i][j]=tempmin
                for i in range(N+1):
                    for j in range(M+1):
                        tempa=float('-inf')
                        for add_i in range(math.floor(V[t]/accuracy_1)+1):
                            if tempa<=state_env[t][min(N,i+add_i)][j]:
                                tempa=state_env[t][min(N,i+add_i)][j]
                                state[t][i][j]=[tempa,add_i]

            inv_discrete=0
            for t in range(T):
                l_idx = min(M, max(0, discrete_L[t]))
                inv_discrete += state[t][inv_discrete][l_idx][1]
            inv_discrete *= accuracy_1
            alg_discrete=p*min(demand,inv_discrete)-c*inv_discrete
            ratio_discrete= alg_discrete/opt
            dis_count=time.perf_counter()-start_our_dis+opt_count

            group_r_dis_1.append(ratio_discrete)
            group_dis_1_time.append(dis_count)

            # discrete optimization 1,2,3,...,T+1 state[T][N][M]=[rgt,a]
            N=math.ceil(sum(V)/accuracy_2) #inv 0,1,...,N
            M=math.ceil(max(max(R),max(L),demand)/accuracy_2) #left 0,1,...,M
            discrete_L=[math.floor(a/accuracy_2) for a in L]
            discrete_R=[math.ceil(a/accuracy_2) for a in R]
            discrete_delta=[x-y for x,y in zip(discrete_R,discrete_L)]
            state=[  [  [  [0,0] for _ in range(M+1)]    for _ in range(N+1)]   for _ in range(T+1)   ]
            state_env=[  [  [  0 for _ in range(M+1)]    for _ in range(N+1)]   for _ in range(T) ]

            start_our_dis=time.perf_counter()
            for i in range(N+1):
                for j in range(1,M+1):
                    state[T][i][j]=[  (p*min(i,j)-c*i) / ( (p-c)*min(N,j) ),0]

            for t in range(T-1,-1,-1):
                for i in range(N+1):
                    for j in range(M+1):
                        tempmin=float('inf')
                        next_delta = discrete_delta[t+1] if t+1 < T else 0
                        for add_j in range(max(discrete_delta[t]-next_delta,0)+1):
                            tempmin=min(tempmin,state[t+1][i][min(M,j+add_j)][0])
                        state_env[t][i][j]=tempmin
                for i in range(N+1):
                    for j in range(M+1):
                        tempa=float('-inf')
                        for add_i in range(math.floor(V[t]/accuracy_2)+1):
                            if tempa<=state_env[t][min(N,i+add_i)][j]:
                                tempa=state_env[t][min(N,i+add_i)][j]
                                state[t][i][j]=[tempa,add_i]

            inv_discrete=0
            for t in range(T):
                l_idx = min(M, max(0, discrete_L[t]))
                inv_discrete += state[t][inv_discrete][l_idx][1]
            inv_discrete *= accuracy_2
            alg_discrete=p*min(demand,inv_discrete)-c*inv_discrete
            ratio_discrete= alg_discrete/opt
            dis_count=time.perf_counter()-start_our_dis+opt_count

            group_r_dis_2.append(ratio_discrete)
            group_dis_2_time.append(dis_count)
        else:
            group_r_dis_1.append(0.0)
            group_dis_1_time.append(0.0)
            group_r_dis_2.append(0.0)
            group_dis_2_time.append(0.0)



    opt_total.append(group_opt)
    r_lw_total.append(group_r_lw)
    r_mid_total.append(group_r_mid)
    r_up_total.append(group_r_up)
    r_dis_1_total.append(group_r_dis_1)
    r_dis_2_total.append(group_r_dis_2)
    lw_time.append(group_lw_time)
    up_time.append(group_up_time)
    mid_time.append(group_mid_time)
    dis_1_time.append(group_dis_1_time)
    dis_2_time.append(group_dis_2_time)

result={'rgt_up':r_up_total,
        'rgt_lw':r_lw_total,
        'rgt_mid':r_mid_total,
        'rgt_dis_1':r_dis_1_total,
        'rgt_dis_2':r_dis_2_total,
        'up_time':up_time,
        'lw_time':lw_time,
        'mid_time':mid_time,
        'dis_1_time':dis_1_time,
        'dis_2_time':dis_2_time,}

with open("results/result_ratio_shift.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False,indent=4)


opt_total=[]
rgt_lw_total=[]
rgt_mid_total=[]
rgt_up_total=[]
rgt_dis_1_total=[]
rgt_dis_2_total=[]
lw_time=[]
up_time=[]
mid_time=[]
dis_1_time=[]
dis_2_time=[]

for T,L_group,R_group,delta_group,demand_group,V_group,c,p in zip(
    data['T'],
    data['L'],
    data['R'],
    data['Delta'],
    data['totaldemand'],
    data['V'],
    data['c'],
    data['p'],
):
    T=T-1

    L_repeats = to_repeat_list(L_group, seed_repeat)
    R_repeats = to_repeat_list(R_group, seed_repeat)
    delta_repeats = to_repeat_list(delta_group, seed_repeat)
    V_repeats = to_repeat_list(V_group, seed_repeat)
    demand_repeats = to_repeat_list(demand_group, seed_repeat)

    group_opt=[]
    group_rgt_lw=[]
    group_rgt_mid=[]
    group_rgt_up=[]
    group_rgt_dis_1=[]
    group_rgt_dis_2=[]
    group_lw_time=[]
    group_up_time=[]
    group_mid_time=[]
    group_dis_1_time=[]
    group_dis_2_time=[]

    for L,R,delta,demand,V in zip(L_repeats, R_repeats, delta_repeats, demand_repeats, V_repeats):
        L=L[:T]
        R=R[:T]
        delta=delta[:T]
        opt = (p-c) * min(demand, sum(V))

        # our method: including two extreme greedy and their mid
        start_opt= time.perf_counter()
        opt_gamma=c*(p-c)/p * max([delta[i]-sum(V[i+1:T]) for i in range(T)])
        opt_count=time.perf_counter()-start_opt
        group_opt.append(opt_gamma)

        inv_mid=0
        inv_lw=0
        inv_up=0

        #mid
        start_our_mid=time.perf_counter()
        for t in range(T):
            a_mid=( R[t]- (opt_gamma)/(p-c) - sum(V[t+1:T])-inv_mid + L[t] +opt_gamma/c - inv_mid   )/2
            a_mid= min(max(0,a_mid),V[t])
            inv_mid+=a_mid

        alg_mid = p*min(demand,inv_mid)-c*inv_mid
        rgt_mid=opt-alg_mid
        our_mid_count=time.perf_counter()-start_our_mid+opt_count

        group_mid_time.append(our_mid_count)
        group_rgt_mid.append(rgt_mid)

        #lw
        start_our_lw=time.perf_counter()
        for t in range(T):
            a_lw= R[t]- (opt_gamma)/(p-c) - sum(V[t+1:T])-inv_lw
            a_lw=min(max(0,a_lw),V[t])
            inv_lw += a_lw

        alg_lw = p*min(demand,inv_lw)-c*inv_lw
        rgt_lw=opt-alg_lw
        our_lw_count=time.perf_counter()-start_our_lw+opt_count

        group_lw_time.append(our_lw_count)
        group_rgt_lw.append(rgt_lw)

        #up
        start_our_up=time.perf_counter()
        for t in range(T):
            a_up= L[t] +opt_gamma/c - inv_up
            a_up=min(max(0,a_up),V[t])
            inv_up+=a_up

        alg_up = p*min(demand,inv_up)-c*inv_up
        rgt_up = opt-alg_up
        our_up_count=time.perf_counter()-start_our_up+opt_count

        group_up_time.append(our_up_count)
        group_rgt_up.append(rgt_up)

        if RUN_DISCRETE_TEST:
            #discrete optimization 1,2,3,...,T+1 state[T][N][M]=[rgt,a]
            N=math.ceil(sum(V)/accuracy_1) #inv 0,1,...,N
            M=math.ceil(max(max(R),max(L),demand)/accuracy_1) #left 0,1,...,M
            discrete_L=[math.floor(a/accuracy_1) for a in L]
            discrete_R=[math.ceil(a/accuracy_1) for a in R]
            discrete_delta=[x-y for x,y in zip(discrete_R,discrete_L)]
            state=[  [  [  [0,0] for _ in range(M+1)]    for _ in range(N+1)]   for _ in range(T+1)   ]
            state_env=[  [  [  [0,0] for _ in range(M+1)]    for _ in range(N+1)]   for _ in range(T) ]

            start_our_dis=time.perf_counter()
            for i in range(N+1):
                for j in range(M+1):
                    state[T][i][j]=[ accuracy_1 * (( (p-c)*min(N,j) )-(p*min(i,j)-c*i)) ,0]

            for t in range(T-1,-1,-1):
                for i in range(N+1):
                    for j in range(M+1):
                        tempmax=0
                        next_delta = discrete_delta[t+1] if t+1 < T else 0
                        for add_j in range(max(discrete_delta[t]-next_delta,0)+1):
                            tempmax=max(tempmax,state[t+1][i][min(M,j+add_j)][0])
                        state_env[t][i][j]=tempmax
                for i in range(N+1):
                    for j in range(M+1):
                        tempa=float('inf')
                        for add_i in range(math.floor(V[t]/accuracy_1)+1):
                            if tempa>=state_env[t][min(N,i+add_i)][j]:
                                tempa=state_env[t][min(N,i+add_i)][j]
                                state[t][i][j]=[tempa,add_i]

            inv_discrete=0
            for t in range(T):
                l_idx = min(M, max(0, discrete_L[t]))
                inv_discrete += state[t][inv_discrete][l_idx][1]
            inv_discrete *= accuracy_1
            alg_discrete=p*min(demand,inv_discrete)-c*inv_discrete
            rgt_discrete= opt-alg_discrete
            dis_count=time.perf_counter()-start_our_dis+opt_count

            group_rgt_dis_1.append(rgt_discrete)
            group_dis_1_time.append(dis_count)


            #discrete optimization 1,2,3,...,T+1 state[T][N][M]=[rgt,a]
            N=math.ceil(sum(V)/accuracy_2) #inv 0,1,...,N
            M=math.ceil(max(max(R),max(L),demand)/accuracy_2) #left 0,1,...,M
            discrete_L=[math.floor(a/accuracy_2) for a in L]
            discrete_R=[math.ceil(a/accuracy_2) for a in R]
            discrete_delta=[x-y for x,y in zip(discrete_R,discrete_L)]
            state=[  [  [  [0,0] for _ in range(M+1)]    for _ in range(N+1)]   for _ in range(T+1)   ]
            state_env=[  [  [  [0,0] for _ in range(M+1)]    for _ in range(N+1)]   for _ in range(T) ]

            start_our_dis=time.perf_counter()
            for i in range(N+1):
                for j in range(M+1):
                    state[T][i][j]=[ accuracy_2 * (( (p-c)*min(N,j) )-(p*min(i,j)-c*i)) ,0]

            for t in range(T-1,-1,-1):
                for i in range(N+1):
                    for j in range(M+1):
                        tempmax=0
                        next_delta = discrete_delta[t+1] if t+1 < T else 0
                        for add_j in range(max(discrete_delta[t]-next_delta,0)+1):
                            tempmax=max(tempmax,state[t+1][i][min(M,j+add_j)][0])
                        state_env[t][i][j]=tempmax
                for i in range(N+1):
                    for j in range(M+1):
                        tempa=float('inf')
                        for add_i in range(math.floor(V[t]/accuracy_2)+1):
                            if tempa>=state_env[t][min(N,i+add_i)][j]:
                                tempa=state_env[t][min(N,i+add_i)][j]
                                state[t][i][j]=[tempa,add_i]

            inv_discrete=0
            for t in range(T):
                l_idx = min(M, max(0, discrete_L[t]))
                inv_discrete += state[t][inv_discrete][l_idx][1]
            inv_discrete *= accuracy_2
            alg_discrete=p*min(demand,inv_discrete)-c*inv_discrete
            rgt_discrete= opt-alg_discrete
            dis_count=time.perf_counter()-start_our_dis+opt_count

            group_rgt_dis_2.append(rgt_discrete)
            group_dis_2_time.append(dis_count)
        else:
            group_rgt_dis_1.append(0.0)
            group_dis_1_time.append(0.0)
            group_rgt_dis_2.append(0.0)
            group_dis_2_time.append(0.0)

    opt_total.append(group_opt)
    rgt_lw_total.append(group_rgt_lw)
    rgt_mid_total.append(group_rgt_mid)
    rgt_up_total.append(group_rgt_up)
    rgt_dis_1_total.append(group_rgt_dis_1)
    rgt_dis_2_total.append(group_rgt_dis_2)
    lw_time.append(group_lw_time)
    up_time.append(group_up_time)
    mid_time.append(group_mid_time)
    dis_1_time.append(group_dis_1_time)
    dis_2_time.append(group_dis_2_time)


result={'rgt_up':rgt_up_total,
        'rgt_lw':rgt_lw_total,
        'rgt_mid':rgt_mid_total,
        'rgt_dis_1':rgt_dis_1_total,
        'rgt_dis_2':rgt_dis_2_total,
        'up_time':up_time,
        'lw_time':lw_time,
        'mid_time':mid_time,
        'dis_1_time':dis_1_time,
        'dis_2_time':dis_2_time,
        'accuracy_1':accuracy_1,
        'accuracy_2':accuracy_2,}

with open("results/result_rgt_shift.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False,indent=4)
