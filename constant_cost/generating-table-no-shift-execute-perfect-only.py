import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

# ===== Perfect-sample generation config =====
num_perfect = 80
base_seed = 10
max_attempts_per_group = 200000

# ===== Execute config =====
accuracy_1 = 1
accuracy_2 = 4


def calibrate_asymmetric_intervals(
    pi,
    estdemand,
    rangedemand,
    n_mc=20000,
    lower_q=0.025,
    upper_q=0.975,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    pi = np.asarray(pi, dtype=float)
    estdemand = np.asarray(estdemand, dtype=float)
    T = len(pi)

    if len(estdemand) != T:
        raise ValueError(f"len(estdemand)={len(estdemand)} must equal len(pi)={T}")
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError("Require 0 <= lower_q < upper_q <= 1")

    err = [[] for _ in range(T)]

    for _ in range(n_mc):
        phi_mc = np.random.binomial(rangedemand, pi, size=T)

        estphi_mc = np.zeros((T, T), dtype=float)
        for t in range(T):
            if t + 1 < T:
                estphi_mc[t, t + 1 :] = np.random.binomial(
                    rangedemand, pi[t + 1 :], size=T - (t + 1)
                )

        for i in range(T):
            true_future = float(phi_mc[i + 1 :].sum())
            if i + 1 < T:
                sample_sum = float(estphi_mc[: i + 1, i + 1 :].sum())
            else:
                sample_sum = 0.0

            sample_mean_future = sample_sum / (i + 1)
            err[i].append(true_future - sample_mean_future)

    q_lower_list = [float(np.quantile(err[i], lower_q)) for i in range(T)]
    q_upper_list = [float(np.quantile(err[i], upper_q)) for i in range(T)]

    L = (estdemand + np.asarray(q_lower_list)).tolist()
    R = (estdemand + np.asarray(q_upper_list)).tolist()

    return q_lower_list, q_upper_list, L, R, upper_q - lower_q


def is_perfect_example(L, R, total_demand):
    demand_value = float(total_demand)
    return all(float(l) <= demand_value <= float(r) for l, r in zip(L, R))


def generate_single_seed_sample(T, maxpi, rangedemand, n_mc, seed):
    pi = np.zeros(T, dtype=float)
    phi = np.zeros(T, dtype=int)

    np.random.seed(seed)
    for i in range(T):
        pi[i] = np.random.uniform(0, maxpi)
        phi[i] = np.random.binomial(rangedemand, pi[i])

    estphi = np.zeros((T, T), dtype=float)
    estdemand = np.zeros(T, dtype=float)

    current_total_demand = 0.0
    current_future_demand = 0.0
    each_demand = np.zeros(T, dtype=float)

    for i in range(T):
        current_total_demand += phi[i]
        for k in range(i + 1, T):
            estphi[i, k] = np.random.binomial(rangedemand, pi[k])
            current_future_demand += estphi[i, k]
            each_demand[k] += estphi[i, k]
        current_future_demand -= each_demand[i]
        estdemand[i] = current_total_demand + current_future_demand / (i + 1)

    _, _, L, R, _ = calibrate_asymmetric_intervals(
        pi=pi,
        estdemand=estdemand,
        rangedemand=rangedemand,
        n_mc=n_mc,
        lower_q=0.05,
        upper_q=0.95,
        seed=seed,
    )

    total_demand = int(phi.sum())
    Delta = [r - l for r, l in zip(R, L)]
    V = [total_demand * 3 * t / (T * (T - 1)) for t in range(T - 1, 0, -1)]

    return {
        "seed": seed,
        "V": V,
        "L": L,
        "R": R,
        "Delta": Delta,
        "totaldemand": total_demand,
        "is_perfect": is_perfect_example(L, R, total_demand),
    }


def generate_dataset_perfect_only(
    T_list,
    maxpi_list,
    rangedemand_list,
    c_list,
    p_list,
    n_mc_list,
    num_perfect,
    base_seed,
    output_path,
    max_attempts_per_group=200000,
):
    V_list = []
    L_list = []
    R_list = []
    Delta_list = []
    demand_list = []
    seed_values_per_group = []
    generation_stats = []

    for group_idx, (T, maxpi, rangedemand, c, p, n_mc) in enumerate(
        zip(T_list, maxpi_list, rangedemand_list, c_list, p_list, n_mc_list)
    ):
        group_V = []
        group_Delta = []
        group_L = []
        group_R = []
        group_demand = []
        group_seeds = []

        attempts = 0
        seed_cursor = base_seed

        while len(group_V) < num_perfect:
            if attempts >= max_attempts_per_group:
                raise RuntimeError(
                    f"Group {group_idx}: reached max attempts={max_attempts_per_group} "
                    f"before collecting num_perfect={num_perfect} perfect samples"
                )

            sample = generate_single_seed_sample(
                T=T,
                maxpi=maxpi,
                rangedemand=rangedemand,
                n_mc=n_mc,
                seed=seed_cursor,
            )
            attempts += 1
            seed_cursor += 1

            if not sample["is_perfect"]:
                continue

            group_V.append(sample["V"])
            group_Delta.append(sample["Delta"])
            group_L.append(sample["L"])
            group_R.append(sample["R"])
            group_demand.append(sample["totaldemand"])
            group_seeds.append(sample["seed"])

        V_list.append(group_V)
        Delta_list.append(group_Delta)
        L_list.append(group_L)
        R_list.append(group_R)
        demand_list.append(group_demand)
        seed_values_per_group.append(group_seeds)
        generation_stats.append(
            {
                "group_index": group_idx,
                "attempts": attempts,
                "num_perfect": num_perfect,
                "acceptance_ratio": num_perfect / attempts,
            }
        )

    data = {
        "V": V_list,
        "c": c_list,
        "p": p_list,
        "rangedemand": rangedemand_list,
        "L": L_list,
        "R": R_list,
        "Delta": Delta_list,
        "T": T_list,
        "totaldemand": demand_list,
        "seed_repeat": num_perfect,
        "seed_values": seed_values_per_group,
        "num_perfect": num_perfect,
        "generation_stats": generation_stats,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return data


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std())


def to_repeat_list(group_value, expected_repeat):
    if expected_repeat <= 0:
        raise ValueError(f"seed_repeat must be positive, got {expected_repeat}")

    if isinstance(group_value, list):
        if len(group_value) == expected_repeat:
            return group_value
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


def ratio_discrete_solver(T, L, R, V, demand, p, c, accuracy, opt, opt_count):
    N = math.ceil(sum(V) / accuracy)
    M = math.ceil(max(max(R), max(L), demand) / accuracy)
    discrete_L = [math.floor(a / accuracy) for a in L]
    discrete_R = [math.ceil(a / accuracy) for a in R]
    discrete_delta = [x - y for x, y in zip(discrete_R, discrete_L)]
    state = [[[[0, 0] for _ in range(M + 1)] for _ in range(N + 1)] for _ in range(T + 1)]
    state_env = [[[0 for _ in range(M + 1)] for _ in range(N + 1)] for _ in range(T)]

    start_dis = time.process_time()
    for i in range(N + 1):
        for j in range(1, M + 1):
            state[T][i][j] = [(p * min(i, j) - c * i) / ((p - c) * min(N, j)), 0]

    for t in range(T - 1, -1, -1):
        for i in range(N + 1):
            for j in range(M + 1):
                tempmin = float("inf")
                next_delta = discrete_delta[t + 1] if t + 1 < T else 0
                for add_j in range(max(discrete_delta[t] - next_delta, 0) + 1):
                    tempmin = min(tempmin, state[t + 1][i][min(M, j + add_j)][0])
                state_env[t][i][j] = tempmin
        for i in range(N + 1):
            for j in range(M + 1):
                tempa = float("-inf")
                for add_i in range(math.floor(V[t] / accuracy) + 1):
                    if tempa <= state_env[t][min(N, i + add_i)][j]:
                        tempa = state_env[t][min(N, i + add_i)][j]
                        state[t][i][j] = [tempa, add_i]

    inv_discrete = 0
    for t in range(T):
        l_idx = min(M, max(0, discrete_L[t]))
        inv_discrete += state[t][inv_discrete][l_idx][1]
    inv_discrete *= accuracy

    alg_discrete = p * min(demand, inv_discrete) - c * inv_discrete
    ratio_discrete = alg_discrete / opt
    dis_time = time.process_time() - start_dis + opt_count
    return ratio_discrete, dis_time


def regret_discrete_solver(T, L, R, V, demand, p, c, accuracy, opt, opt_count):
    N = math.ceil(sum(V) / accuracy)
    M = math.ceil(max(max(R), max(L), demand) / accuracy)
    discrete_L = [math.floor(a / accuracy) for a in L]
    discrete_R = [math.ceil(a / accuracy) for a in R]
    discrete_delta = [x - y for x, y in zip(discrete_R, discrete_L)]
    state = [[[[0, 0] for _ in range(M + 1)] for _ in range(N + 1)] for _ in range(T + 1)]
    state_env = [[[[0, 0] for _ in range(M + 1)] for _ in range(N + 1)] for _ in range(T)]

    start_dis = time.process_time()
    for i in range(N + 1):
        for j in range(M + 1):
            state[T][i][j] = [accuracy * (((p - c) * min(N, j)) - (p * min(i, j) - c * i)), 0]

    for t in range(T - 1, -1, -1):
        for i in range(N + 1):
            for j in range(M + 1):
                tempmax = 0
                next_delta = discrete_delta[t + 1] if t + 1 < T else 0
                for add_j in range(max(discrete_delta[t] - next_delta, 0) + 1):
                    tempmax = max(tempmax, state[t + 1][i][min(M, j + add_j)][0])
                state_env[t][i][j] = tempmax
        for i in range(N + 1):
            for j in range(M + 1):
                tempa = float("inf")
                for add_i in range(math.floor(V[t] / accuracy) + 1):
                    if tempa >= state_env[t][min(N, i + add_i)][j]:
                        tempa = state_env[t][min(N, i + add_i)][j]
                        state[t][i][j] = [tempa, add_i]

    inv_discrete = 0
    for t in range(T):
        l_idx = min(M, max(0, discrete_L[t]))
        inv_discrete += state[t][inv_discrete][l_idx][1]
    inv_discrete *= accuracy

    alg_discrete = p * min(demand, inv_discrete) - c * inv_discrete
    rgt_discrete = opt - alg_discrete
    dis_time = time.process_time() - start_dis + opt_count
    return rgt_discrete, dis_time


def LP_base_regret_solver(T, L, R, V, demand, p, c, opt):
    start_lp = time.process_time()
    T = T - 1
    delta = [R[t] - L[t] for t in range(T)]
    lp_C = c
    lp_c = p - c
    sol_c = [0] * T + [1]
    sol_A = []
    sol_b = []
    for t in range(T):
        row = [1] * (t + 1) + [0] * (T - t - 1) + [-1 / lp_C]
        sol_A.append(row)
        sol_b.append(R[0] - min(delta[: (t + 1)]))
    sol_A.append([-1] * T + [-1 / lp_c])
    sol_b.append(-R[0])
    sol_bounds = [(0, V[t]) for t in range(T)] + [(0, None)]
    res = linprog(c=sol_c, A_ub=sol_A, b_ub=sol_b, bounds=sol_bounds, method="highs")
    LP_x = res.x

    inv = 0
    for t in range(T):
        add = sum(LP_x[: (t + 1)]) - inv - (R[0] - min(R[: (t + 1)]))
        add = max(0, add)
        add = min(add, V[t])
        inv += add

    lp_time = time.process_time() - start_lp
    regret_lp = opt - (p * min(demand, inv) - c * inv)
    return regret_lp, lp_time


def simulate_naive_greedy_inventory(T, L, R, V, p, c):
    inv = 0.0
    for t in range(T):
        target = (((p - c) * R[t]) + (c * L[t])) / p
        add = target - inv
        add = min(max(0.0, add), V[t])
        inv += add
    return inv


def run_ratio_and_regret(dataset, run_discrete=True):
    seed_repeat = dataset.get("seed_repeat", 1)

    ratio_r_lw_total = []
    ratio_r_mid_total = []
    ratio_r_up_total = []
    ratio_r_dis_1_total = []
    ratio_r_dis_2_total = []
    ratio_r_naive_greedy_total = []
    ratio_lw_time = []
    ratio_up_time = []
    ratio_mid_time = []
    ratio_dis_1_time = []
    ratio_dis_2_time = []
    ratio_naive_greedy_time = []

    rgt_lw_total = []
    rgt_mid_total = []
    rgt_up_total = []
    rgt_dis_1_total = []
    rgt_dis_2_total = []
    rgt_lp_total = []
    rgt_naive_greedy_total = []
    rgt_lw_time = []
    rgt_up_time = []
    rgt_mid_time = []
    rgt_dis_1_time = []
    rgt_dis_2_time = []
    rgt_lp_time = []
    rgt_naive_greedy_time = []

    for T, L_group, R_group, delta_group, demand_group, V_group, c, p in zip(
        dataset["T"],
        dataset["L"],
        dataset["R"],
        dataset["Delta"],
        dataset["totaldemand"],
        dataset["V"],
        dataset["c"],
        dataset["p"],
    ):
        T = T - 1

        L_repeats = to_repeat_list(L_group, seed_repeat)
        R_repeats = to_repeat_list(R_group, seed_repeat)
        delta_repeats = to_repeat_list(delta_group, seed_repeat)
        V_repeats = to_repeat_list(V_group, seed_repeat)
        demand_repeats = to_repeat_list(demand_group, seed_repeat)

        group_ratio_r_lw = []
        group_ratio_r_mid = []
        group_ratio_r_up = []
        group_ratio_r_dis_1 = []
        group_ratio_r_dis_2 = []
        group_ratio_r_naive_greedy = []
        group_ratio_lw_time = []
        group_ratio_up_time = []
        group_ratio_mid_time = []
        group_ratio_dis_1_time = []
        group_ratio_dis_2_time = []
        group_ratio_naive_greedy_time = []

        group_rgt_lw = []
        group_rgt_mid = []
        group_rgt_up = []
        group_rgt_dis_1 = []
        group_rgt_dis_2 = []
        group_rgt_lp = []
        group_rgt_naive_greedy = []
        group_rgt_lw_time = []
        group_rgt_up_time = []
        group_rgt_mid_time = []
        group_rgt_dis_1_time = []
        group_rgt_dis_2_time = []
        group_rgt_lp_time = []
        group_rgt_naive_greedy_time = []

        for L, R, delta, demand, V in zip(
            L_repeats, R_repeats, delta_repeats, demand_repeats, V_repeats
        ):
            L = L[:T]
            R = R[:T]
            delta = delta[:T]
            opt = (p - c) * min(demand, sum(V))

            start_opt = time.process_time()
            opt_gamma_ratio = min(
                [
                    (p * L[0] + c * sum(V[i + 1 : T]))
                    / (p * L[0] + c * delta[i])
                    for i in range(T)
                ]
            )
            opt_count_ratio = time.process_time() - start_opt

            inv_mid = 0
            inv_lw = 0
            inv_up = 0

            start_mid = time.process_time()
            for t in range(T):
                a_mid = (
                    R[t] * opt_gamma_ratio
                    - sum(V[t + 1 : T])
                    + L[t] * (((1 - opt_gamma_ratio) * p + opt_gamma_ratio * c) / c)
                ) / 2 - inv_mid
                a_mid = min(max(0, a_mid), V[t])
                inv_mid += a_mid
            alg_mid = p * min(demand, inv_mid) - c * inv_mid
            ratio_mid = alg_mid / opt
            group_ratio_mid_time.append(time.process_time() - start_mid + opt_count_ratio)
            group_ratio_r_mid.append(ratio_mid)

            start_lw = time.process_time()
            for t in range(T):
                a_lw = R[t] * opt_gamma_ratio - sum(V[t + 1 : T]) - inv_lw
                a_lw = min(max(0, a_lw), V[t])
                inv_lw += a_lw
            alg_lw = p * min(demand, inv_lw) - c * inv_lw
            ratio_lw = alg_lw / opt
            group_ratio_lw_time.append(time.process_time() - start_lw + opt_count_ratio)
            group_ratio_r_lw.append(ratio_lw)

            start_up = time.process_time()
            for t in range(T):
                a_up = L[t] * (((1 - opt_gamma_ratio) * p + opt_gamma_ratio * c) / c) - inv_up
                a_up = min(max(0, a_up), V[t])
                inv_up += a_up
            alg_up = p * min(demand, inv_up) - c * inv_up
            ratio_up = alg_up / opt
            group_ratio_up_time.append(time.process_time() - start_up + opt_count_ratio)
            group_ratio_r_up.append(ratio_up)

            if run_discrete:
                ratio_dis_1, dis_1_time = ratio_discrete_solver(
                    T, L, R, V, demand, p, c, accuracy_1, opt, opt_count_ratio
                )
                ratio_dis_2, dis_2_time = ratio_discrete_solver(
                    T, L, R, V, demand, p, c, accuracy_2, opt, opt_count_ratio
                )
            else:
                ratio_dis_1, dis_1_time = 0.0, 0.0
                ratio_dis_2, dis_2_time = 0.0, 0.0
            group_ratio_r_dis_1.append(ratio_dis_1)
            group_ratio_r_dis_2.append(ratio_dis_2)
            group_ratio_dis_1_time.append(dis_1_time)
            group_ratio_dis_2_time.append(dis_2_time)

            start_opt = time.process_time()
            opt_gamma_rgt = c * (p - c) / p * max(
                [delta[i] - sum(V[i + 1 : T]) for i in range(T)]
            )
            opt_count_rgt = time.process_time() - start_opt

            inv_mid = 0
            inv_lw = 0
            inv_up = 0

            start_mid = time.process_time()
            for t in range(T):
                a_mid = (
                    R[t]
                    - (opt_gamma_rgt) / (p - c)
                    - sum(V[t + 1 : T])
                    - inv_mid
                    + L[t]
                    + opt_gamma_rgt / c
                    - inv_mid
                ) / 2
                a_mid = min(max(0, a_mid), V[t])
                inv_mid += a_mid
            alg_mid = p * min(demand, inv_mid) - c * inv_mid
            rgt_mid = opt - alg_mid
            group_rgt_mid_time.append(time.process_time() - start_mid + opt_count_rgt)
            group_rgt_mid.append(rgt_mid)

            start_lw = time.process_time()
            for t in range(T):
                a_lw = R[t] - (opt_gamma_rgt) / (p - c) - sum(V[t + 1 : T]) - inv_lw
                a_lw = min(max(0, a_lw), V[t])
                inv_lw += a_lw
            alg_lw = p * min(demand, inv_lw) - c * inv_lw
            rgt_lw = opt - alg_lw
            group_rgt_lw_time.append(time.process_time() - start_lw + opt_count_rgt)
            group_rgt_lw.append(rgt_lw)

            start_up = time.process_time()
            for t in range(T):
                a_up = L[t] + opt_gamma_rgt / c - inv_up
                a_up = min(max(0, a_up), V[t])
                inv_up += a_up
            alg_up = p * min(demand, inv_up) - c * inv_up
            rgt_up = opt - alg_up
            group_rgt_up_time.append(time.process_time() - start_up + opt_count_rgt)
            group_rgt_up.append(rgt_up)

            if run_discrete:
                rgt_dis_1, dis_1_elapsed = regret_discrete_solver(
                    T, L, R, V, demand, p, c, accuracy_1, opt, opt_count_rgt
                )
                rgt_dis_2, dis_2_elapsed = regret_discrete_solver(
                    T, L, R, V, demand, p, c, accuracy_2, opt, opt_count_rgt
                )
            else:
                rgt_dis_1, dis_1_elapsed = 0.0, 0.0
                rgt_dis_2, dis_2_elapsed = 0.0, 0.0
            group_rgt_dis_1.append(rgt_dis_1)
            group_rgt_dis_2.append(rgt_dis_2)
            group_rgt_dis_1_time.append(dis_1_elapsed)
            group_rgt_dis_2_time.append(dis_2_elapsed)

            rgt_lp, lp_elapsed = LP_base_regret_solver(T, L, R, V, demand, p, c, opt)
            group_rgt_lp.append(rgt_lp)
            group_rgt_lp_time.append(lp_elapsed)

            start_greedy = time.process_time()
            inv_greedy = simulate_naive_greedy_inventory(T, L, R, V, p, c)
            greedy_elapsed = time.process_time() - start_greedy
            alg_greedy = p * min(demand, inv_greedy) - c * inv_greedy
            ratio_greedy = alg_greedy / opt if abs(opt) > 1e-12 else 0.0
            rgt_greedy = opt - alg_greedy

            group_ratio_r_naive_greedy.append(ratio_greedy)
            group_ratio_naive_greedy_time.append(greedy_elapsed)
            group_rgt_naive_greedy.append(rgt_greedy)
            group_rgt_naive_greedy_time.append(greedy_elapsed)

        ratio_r_lw_total.append(group_ratio_r_lw)
        ratio_r_mid_total.append(group_ratio_r_mid)
        ratio_r_up_total.append(group_ratio_r_up)
        ratio_r_dis_1_total.append(group_ratio_r_dis_1)
        ratio_r_dis_2_total.append(group_ratio_r_dis_2)
        ratio_r_naive_greedy_total.append(group_ratio_r_naive_greedy)
        ratio_lw_time.append(group_ratio_lw_time)
        ratio_up_time.append(group_ratio_up_time)
        ratio_mid_time.append(group_ratio_mid_time)
        ratio_dis_1_time.append(group_ratio_dis_1_time)
        ratio_dis_2_time.append(group_ratio_dis_2_time)
        ratio_naive_greedy_time.append(group_ratio_naive_greedy_time)

        rgt_lw_total.append(group_rgt_lw)
        rgt_mid_total.append(group_rgt_mid)
        rgt_up_total.append(group_rgt_up)
        rgt_dis_1_total.append(group_rgt_dis_1)
        rgt_dis_2_total.append(group_rgt_dis_2)
        rgt_lp_total.append(group_rgt_lp)
        rgt_naive_greedy_total.append(group_rgt_naive_greedy)
        rgt_lw_time.append(group_rgt_lw_time)
        rgt_up_time.append(group_rgt_up_time)
        rgt_mid_time.append(group_rgt_mid_time)
        rgt_dis_1_time.append(group_rgt_dis_1_time)
        rgt_dis_2_time.append(group_rgt_dis_2_time)
        rgt_lp_time.append(group_rgt_lp_time)
        rgt_naive_greedy_time.append(group_rgt_naive_greedy_time)

    ratio_methods = {
        "lw": (ratio_r_lw_total, ratio_lw_time),
        "mid": (ratio_r_mid_total, ratio_mid_time),
        "up": (ratio_r_up_total, ratio_up_time),
        "dis_1": (ratio_r_dis_1_total, ratio_dis_1_time),
        "dis_2": (ratio_r_dis_2_total, ratio_dis_2_time),
        "naive_greedy": (ratio_r_naive_greedy_total, ratio_naive_greedy_time),
    }

    regret_methods = {
        "lw": (rgt_lw_total, rgt_lw_time),
        "mid": (rgt_mid_total, rgt_mid_time),
        "up": (rgt_up_total, rgt_up_time),
        "dis_1": (rgt_dis_1_total, rgt_dis_1_time),
        "dis_2": (rgt_dis_2_total, rgt_dis_2_time),
        "lp": (rgt_lp_total, rgt_lp_time),
        "naive_greedy": (rgt_naive_greedy_total, rgt_naive_greedy_time),
    }

    ratio_summary = {}
    for method, (metric_groups, time_groups) in ratio_methods.items():
        method_summary = []
        for metric_values, time_values in zip(metric_groups, time_groups):
            metric_mean, metric_std = mean_std(metric_values)
            time_mean, time_std = mean_std(time_values)
            method_summary.append(
                {
                    "ratio_mean": metric_mean,
                    "ratio_std": metric_std,
                    "time_mean": time_mean,
                    "time_std": time_std,
                }
            )
        ratio_summary[method] = method_summary

    regret_summary = {}
    for method, (metric_groups, time_groups) in regret_methods.items():
        method_summary = []
        for metric_values, time_values in zip(metric_groups, time_groups):
            metric_mean, metric_std = mean_std(metric_values)
            time_mean, time_std = mean_std(time_values)
            method_summary.append(
                {
                    "regret_mean": metric_mean,
                    "regret_std": metric_std,
                    "time_mean": time_mean,
                    "time_std": time_std,
                }
            )
        regret_summary[method] = method_summary

    ratio_result = {
        "seed_repeat": seed_repeat,
        "num_perfect": dataset.get("num_perfect", seed_repeat),
        "T": dataset["T"],
        "accuracy_1": accuracy_1,
        "accuracy_2": accuracy_2,
        "methods": ratio_summary,
    }

    rgt_result = {
        "seed_repeat": seed_repeat,
        "num_perfect": dataset.get("num_perfect", seed_repeat),
        "T": dataset["T"],
        "accuracy_1": accuracy_1,
        "accuracy_2": accuracy_2,
        "methods": regret_summary,
    }

    return ratio_result, rgt_result


def main():
    w = 1
    T_list = [200] * w
    maxpi_list = [0.5] * w
    rangedemand_list = [5] * w
    c_list = [1] * w
    p_list = [2] * w
    n_mc_list = [100] * w

    generate_output = RESULTS_DIR / "generate_table_perfect_only.json"
    ratio_output = RESULTS_DIR / "result_ratio_table_perfect_only.json"
    rgt_output = RESULTS_DIR / "result_rgt_table_perfect_only.json"

    data = generate_dataset_perfect_only(
        T_list=T_list,
        maxpi_list=maxpi_list,
        rangedemand_list=rangedemand_list,
        c_list=c_list,
        p_list=p_list,
        n_mc_list=n_mc_list,
        num_perfect=num_perfect,
        base_seed=base_seed,
        output_path=generate_output,
        max_attempts_per_group=max_attempts_per_group,
    )

    ratio_result, rgt_result = run_ratio_and_regret(data, run_discrete=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ratio_output, "w", encoding="utf-8") as f:
        json.dump(ratio_result, f, ensure_ascii=False, indent=4)

    with open(rgt_output, "w", encoding="utf-8") as f:
        json.dump(rgt_result, f, ensure_ascii=False, indent=4)

    print(f"Saved perfect-only dataset to: {generate_output}")
    print(f"Saved ratio result to: {ratio_output}")
    print(f"Saved regret result to: {rgt_output}")


if __name__ == "__main__":
    main()
