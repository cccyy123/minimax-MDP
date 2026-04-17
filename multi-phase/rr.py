import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import phase_reduction

# Maximum number of CPU cores to use; <=0 means use all available cores.
max_cores = 70


# Measure CPU time (process time) instead of wall-clock elapsed time.
_timer = time.process_time


def _normalize_repeat_values(group_value, repeat):
    if isinstance(group_value, list) and len(group_value) == repeat:
        return group_value
    return [group_value for _ in range(repeat)]


def expand_seed_runs_with_group_sizes(dataset):
    n_groups = len(dataset.get("K_list", []))
    expanded = {key: [] for key in dataset.keys()}
    group_sizes = []
    per_seed_fields = {"L", "R", "Delta", "V", "totaldemand", "seed_list"}

    for i in range(n_groups):
        repeat_candidates = []

        seed_group = dataset.get("seed_list", [None] * n_groups)[i]
        if isinstance(seed_group, list) and len(seed_group) > 0:
            repeat_candidates.append(len(seed_group))

        for field in ("L", "R", "V", "Delta"):
            group_value = dataset.get(field, [None] * n_groups)[i]
            if isinstance(group_value, list) and len(group_value) > 0 and isinstance(group_value[0], list):
                repeat_candidates.append(len(group_value))

        demand_group = dataset.get("totaldemand", [None] * n_groups)[i]
        if isinstance(demand_group, list) and len(demand_group) > 0 and not isinstance(demand_group[0], list):
            repeat_candidates.append(len(demand_group))

        repeat = max(repeat_candidates) if repeat_candidates else 1
        group_sizes.append(repeat)

        normalized = {}
        for field in per_seed_fields:
            if field in dataset:
                normalized[field] = _normalize_repeat_values(dataset[field][i], repeat)

        for r in range(repeat):
            for key, values in dataset.items():
                value = values[i]
                if key in normalized:
                    expanded[key].append(normalized[key][r])
                else:
                    expanded[key].append(value)

    return expanded, group_sizes


def regroup_by_group_sizes(flat_values, group_sizes):
    grouped_values = []
    start = 0
    for size in group_sizes:
        grouped_values.append(flat_values[start : start + size])
        start += size
    return grouped_values


def _resolve_num_workers():
    available = os.cpu_count() or 1
    if max_cores > 0:
        return min(max_cores, available)
    return available


def _compute_ratio_single(sample, use_true_partitions):
    (
        K,
        accuracy_func,
        accuracy_ratio,
        tau,
        gamma,
        p,
        lw_d,
        up_d,
        _T,
        d,
        V,
        true_K,
        true_tau_K,
        true_gamma,
    ) = sample

    if use_true_partitions:
        K = true_K
        tau = true_tau_K
        gamma = true_gamma

    start_time = _timer()
    delta = [y - x for x, y in zip(lw_d, up_d)]

    n_ratio = 1 + math.floor(1 / accuracy_ratio)
    n_func = math.ceil(max(up_d) / accuracy_func) + 1

    opt = p * d
    tempdemand = d
    for k in range(true_K):
        usedemand = min(sum(V[true_tau_K[k] : true_tau_K[k + 1]]), tempdemand)
        opt -= true_gamma[k] * usedemand
        tempdemand -= usedemand

    optimal_reward = []
    batch_V = [0] * K
    for i in range(K):
        batch_V[i] = sum(V[tau[i] : tau[i + 1]])

    for i in range(n_func):
        demand = i * accuracy_func
        reward = 0
        k = 0
        while k < K:
            reward += min(batch_V[k], demand) * (p - gamma[k])
            demand -= min(batch_V[k], demand)
            if demand != 0:
                k += 1
            else:
                break
        optimal_reward.append(reward)

    phase_inner_iterations_total = 0
    for j in range(n_ratio, -1, -1):
        total_coefficient_L = []
        total_coefficient_R = []
        total_L = []
        total_R = []

        total_coefficient_L.append([[p - x for x in gamma]])
        total_coefficient_R.append(gamma)

        phi = j * accuracy_ratio

        total_R.append([p * i * accuracy_func - phi * optimal_reward[i] for i in range(n_func)])
        total_L.append([[phi * optimal_reward[i] for i in range(n_func)]])

        for k in range(K, 0, -1):
            new_matrix_L, new_L, new_matrix_R, inner_loop_iterations = phase_reduction.phase_reduction(
                diff_tau=tau[k] - tau[k - 1],
                matrix_L=total_coefficient_L[-1],
                L=total_L[-1],
                matrix_R=total_coefficient_R[-1],
                R=total_R[-1],
                V=V[tau[k - 1] : tau[k]],
                delta=delta[tau[k - 1] : tau[k] + 1],
                accuracy=accuracy_func,
            )
            phase_inner_iterations_total += inner_loop_iterations
            total_coefficient_L.append(new_matrix_L)
            total_L.append(new_L)
            total_coefficient_R.append(new_matrix_R)
            total_R.append([p * i * accuracy_func - phi * optimal_reward[i] for i in range(n_func)])
        new_L = np.array(new_L)
        if new_L[:, math.ceil(lw_d[0] / accuracy_func)].max() <= 0 and 0 <= total_R[-1][math.ceil(lw_d[0] / accuracy_func)]:
            break

    opt_time = _timer() - start_time

    start_time = _timer()
    A = []
    buying_plan = [0] * K
    for k in range(K):
        cost = np.dot(np.array(gamma), np.array(buying_plan))
        inv_current = 0
        for t in range(tau[k], tau[k + 1]):
            idx_raw = lw_d[t] / accuracy_func
            idx = int(math.floor(idx_raw))
            A.append(min(max(0, (p * lw_d[t] - phi * optimal_reward[idx] - cost) / gamma[k] - inv_current), V[t]))
            inv_current += A[-1]

        buying_plan[k] = inv_current
    cost = 0
    for k in range(true_K):
        for t in range(true_tau_K[k], true_tau_K[k + 1]):
            cost += true_gamma[k] * A[t]

    reward = p * min(d, sum(A)) - cost

    time_up = _timer() - start_time + opt_time
    perf_up = reward / opt

    start_time = _timer()
    buying_plan = [0] * K
    A = []
    for k in range(K):
        pre_value = []
        for x in total_coefficient_L[-2 - k]:
            pre_value.append(np.dot(np.array(x), np.array(buying_plan)[: k + 1]))
        current_L = (np.array(total_L[-2 - k]) - np.array(pre_value)[:, None]) / np.array(np.array(total_coefficient_L[-2 - k])[:, -1])[:, None]
        current_L = current_L.max(axis=0)
        inv_current = 0
        for t in range(tau[k], tau[k + 1]):
            temp = current_L[int(lw_d[t] // accuracy_func) : int(1 + (lw_d[t] + max(0, delta[t] - delta[tau[k + 1]])) // accuracy_func)]
            A.append(min(max(0, temp.max() - inv_current - sum(V[t + 1 : tau[k + 1]])), V[t]))
            inv_current += A[-1]

        buying_plan[k] = inv_current

    cost = 0
    for k in range(true_K):
        for t in range(true_tau_K[k], true_tau_K[k + 1]):
            cost += true_gamma[k] * A[t]

    reward = p * min(d, sum(A)) - cost

    time_lw = _timer() - start_time + opt_time
    perf_lw = reward / opt

    start_time = _timer()
    buying_plan = [0] * K
    A = []
    for k in range(K):
        cost = np.dot(np.array(gamma), np.array(buying_plan))
        inv_current = 0
        pre_value = []
        for x in total_coefficient_L[-2 - k]:
            pre_value.append(np.dot(np.array(x), np.array(buying_plan)[: k + 1]))
        current_L = (np.array(total_L[-2 - k]) - np.array(pre_value)[:, None]) / np.array(np.array(total_coefficient_L[-2 - k])[:, -1])[:, None]
        current_L = current_L.max(axis=0)
        for t in range(tau[k], tau[k + 1]):
            temp = current_L[int(lw_d[t] // accuracy_func) : int(1 + (lw_d[t] + max(0, delta[t] - delta[tau[k + 1]])) // accuracy_func)]
            add1 = min(max(0, temp.max() - inv_current - sum(V[t + 1 : tau[k + 1]])), V[t])
            add2 = min(max(0, (p * lw_d[t] - phi * optimal_reward[int(lw_d[t] // accuracy_func)] - cost) / gamma[k] - inv_current), V[t])
            A.append((add1 + add2) / 2)
            inv_current += A[-1]

        buying_plan[k] = inv_current

    cost = 0
    for k in range(true_K):
        for t in range(true_tau_K[k], true_tau_K[k + 1]):
            cost += true_gamma[k] * A[t]
    reward = p * min(d, sum(A)) - cost

    time_mid = _timer() - start_time + opt_time
    perf_mid = reward / opt

    return (
        time_up,
        perf_up,
        time_lw,
        perf_lw,
        time_mid,
        perf_mid,
        phase_inner_iterations_total,
    )


def _compute_regret_single(sample, use_true_partitions):
    (
        K,
        accuracy_func,
        accuracy_regret,
        tau,
        gamma,
        p,
        lw_d,
        up_d,
        _T,
        d,
        V,
        true_K,
        true_tau_K,
        true_gamma,
    ) = sample

    if use_true_partitions:
        K = true_K
        tau = true_tau_K
        gamma = true_gamma

    start_time = _timer()
    delta = [y - x for x, y in zip(lw_d, up_d)]

    n_func = math.ceil(max(up_d) / accuracy_func) + 1

    opt = p * d
    tempdemand = d
    for k in range(true_K):
        usedemand = min(sum(V[true_tau_K[k] : true_tau_K[k + 1]]), tempdemand)
        opt -= true_gamma[k] * usedemand
        tempdemand -= usedemand

    optimal_reward = []
    batch_V = [0] * K
    for i in range(K):
        batch_V[i] = sum(V[tau[i] : tau[i + 1]])

    for i in range(n_func):
        demand = i * accuracy_func
        reward = 0
        k = 0
        while k < K:
            reward += min(batch_V[k], demand) * (p - gamma[k])
            demand -= min(batch_V[k], demand)
            if demand != 0:
                k += 1
            else:
                break
        optimal_reward.append(reward)

    n_regret = 1 + math.floor(optimal_reward[min(1 + int(sum(V) // accuracy_func), len(optimal_reward) - 1)] / accuracy_regret)

    phase_inner_iterations_total = 0
    for j in range(0, n_regret + 1):
        total_coefficient_L = []
        total_coefficient_R = []
        total_L = []
        total_R = []

        total_coefficient_L.append([[p - x for x in gamma]])
        total_coefficient_R.append(gamma)

        phi = j * accuracy_regret

        total_R.append([p * i * accuracy_func - (optimal_reward[i] - phi) for i in range(n_func)])
        total_L.append([[optimal_reward[i] - phi for i in range(n_func)]])

        for k in range(K, 0, -1):
            new_matrix_L, new_L, new_matrix_R, inner_loop_iterations = phase_reduction.phase_reduction(
                diff_tau=tau[k] - tau[k - 1],
                matrix_L=total_coefficient_L[-1],
                L=total_L[-1],
                matrix_R=total_coefficient_R[-1],
                R=total_R[-1],
                V=V[tau[k - 1] : tau[k]],
                delta=delta[tau[k - 1] : tau[k] + 1],
                accuracy=accuracy_func,
            )
            phase_inner_iterations_total += inner_loop_iterations
            total_coefficient_L.append(new_matrix_L)
            total_L.append(new_L)
            total_coefficient_R.append(new_matrix_R)
            total_R.append([p * i * accuracy_func - (optimal_reward[i] - phi) for i in range(n_func)])
        new_L = np.array(new_L)
        idx_check = math.ceil(lw_d[0] / accuracy_func)
        cond_left = new_L[:, idx_check].max()
        cond_right = total_R[-1][idx_check]
        if cond_left <= 0 and 0 <= cond_right:
            break

    opt_time = _timer() - start_time

    start_time = _timer()
    buying_plan = [0] * K
    A = []
    for k in range(K):
        cost = np.dot(np.array(gamma), np.array(buying_plan))
        inv_current = 0
        for t in range(tau[k], tau[k + 1]):
            idx_raw = lw_d[t] / accuracy_func
            idx = int(math.floor(idx_raw))
            A.append(min(max(0, (p * lw_d[t] - (optimal_reward[idx] - phi) - cost) / gamma[k] - inv_current), V[t]))
            inv_current += A[-1]

        buying_plan[k] = inv_current

    cost = 0
    for k in range(true_K):
        for t in range(true_tau_K[k], true_tau_K[k + 1]):
            cost += true_gamma[k] * A[t]

    reward = p * min(d, sum(A)) - cost

    time_up = _timer() - start_time + opt_time
    perf_up = -reward + opt

    start_time = _timer()
    buying_plan = [0] * K
    A = []
    for k in range(K):
        pre_value = []
        for x in total_coefficient_L[-2 - k]:
            pre_value.append(np.dot(np.array(x), np.array(buying_plan)[: k + 1]))
        current_L = (np.array(total_L[-2 - k]) - np.array(pre_value)[:, None]) / np.array(np.array(total_coefficient_L[-2 - k])[:, -1])[:, None]
        current_L = current_L.max(axis=0)
        inv_current = 0
        for t in range(tau[k], tau[k + 1]):
            temp = current_L[int(lw_d[t] // accuracy_func) : int(1 + (lw_d[t] + max(0, delta[t] - delta[tau[k + 1]])) // accuracy_func)]
            A.append(min(max(0, temp.max() - inv_current - sum(V[t + 1 : tau[k + 1]])), V[t]))
            inv_current += A[-1]

        buying_plan[k] = inv_current

    cost = 0
    for k in range(true_K):
        for t in range(true_tau_K[k], true_tau_K[k + 1]):
            cost += true_gamma[k] * A[t]

    reward = p * min(d, sum(A)) - cost

    time_lw = _timer() - start_time + opt_time
    perf_lw = -reward + opt

    start_time = _timer()
    buying_plan = [0] * K
    A = []
    for k in range(K):
        cost = np.dot(np.array(gamma), np.array(buying_plan))
        inv_current = 0
        pre_value = []
        for x in total_coefficient_L[-2 - k]:
            pre_value.append(np.dot(np.array(x), np.array(buying_plan)[: k + 1]))
        current_L = (np.array(total_L[-2 - k]) - np.array(pre_value)[:, None]) / np.array(np.array(total_coefficient_L[-2 - k])[:, -1])[:, None]
        current_L = current_L.max(axis=0)
        for t in range(tau[k], tau[k + 1]):
            temp = current_L[int(lw_d[t] // accuracy_func) : int(1 + (lw_d[t] + max(0, delta[t] - delta[tau[k + 1]])) // accuracy_func)]
            add1 = min(max(0, temp.max() - inv_current - sum(V[t + 1 : tau[k + 1]])), V[t])
            add2 = min(max(0, (p * lw_d[t] - (optimal_reward[int(lw_d[t] // accuracy_func)] - phi) - cost) / gamma[k] - inv_current), V[t])
            A.append((add1 + add2) / 2)
            inv_current += A[-1]

        buying_plan[k] = inv_current

    cost = 0
    for k in range(true_K):
        for t in range(true_tau_K[k], true_tau_K[k + 1]):
            cost += true_gamma[k] * A[t]

    reward = p * min(d, sum(A)) - cost

    time_mid = _timer() - start_time + opt_time
    perf_mid = -reward + opt

    return (
        time_up,
        perf_up,
        time_lw,
        perf_lw,
        time_mid,
        perf_mid,
        phase_inner_iterations_total,
    )


def _run_ratio_experiment(data, group_sizes, use_true_partitions, output_path, num_workers):
    samples = list(
        zip(
            data["K_list"],
            data["accuracy_func_list"],
            data["accuracy_ratio_list"],
            data["tau_K_list"],
            data["gamma"],
            data["p"],
            data["L"],
            data["R"],
            data["T"],
            data["totaldemand"],
            data["V"],
            data["true_K_list"],
            data["true_tau_K_list"],
            data["true_gamma_list"],
        )
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_ratio_worker_true if use_true_partitions else _ratio_worker_false, samples))

    time_list_up = [x[0] for x in results]
    performance_list_up = [x[1] for x in results]
    time_list_lw = [x[2] for x in results]
    performance_list_lw = [x[3] for x in results]
    time_list_mid = [x[4] for x in results]
    performance_list_mid = [x[5] for x in results]
    phase_inner_loop_count = [x[6] for x in results]

    data_result = {
        "time_list_up": regroup_by_group_sizes(time_list_up, group_sizes),
        "time_list_lw": regroup_by_group_sizes(time_list_lw, group_sizes),
        "time_list_mid": regroup_by_group_sizes(time_list_mid, group_sizes),
        "accuracy_func_list": data["accuracy_func_list"],
        "accuracy_ratio_list": data["accuracy_ratio_list"],
        "performance_list_up": regroup_by_group_sizes(performance_list_up, group_sizes),
        "performance_list_lw": regroup_by_group_sizes(performance_list_lw, group_sizes),
        "performance_list_mid": regroup_by_group_sizes(performance_list_mid, group_sizes),
        "phase_inner_loop_count": regroup_by_group_sizes(phase_inner_loop_count, group_sizes),
        "group_sizes": group_sizes,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_result, f, ensure_ascii=False, indent=4)


def _run_regret_experiment(data, group_sizes, use_true_partitions, output_path, num_workers):
    samples = list(
        zip(
            data["K_list"],
            data["accuracy_func_list"],
            data["accuracy_regret_list"],
            data["tau_K_list"],
            data["gamma"],
            data["p"],
            data["L"],
            data["R"],
            data["T"],
            data["totaldemand"],
            data["V"],
            data["true_K_list"],
            data["true_tau_K_list"],
            data["true_gamma_list"],
        )
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_regret_worker_true if use_true_partitions else _regret_worker_false, samples))

    time_list_up = [x[0] for x in results]
    performance_list_up = [x[1] for x in results]
    time_list_lw = [x[2] for x in results]
    performance_list_lw = [x[3] for x in results]
    time_list_mid = [x[4] for x in results]
    performance_list_mid = [x[5] for x in results]
    phase_inner_loop_count = [x[6] for x in results]

    data_result = {
        "time_list_up": regroup_by_group_sizes(time_list_up, group_sizes),
        "time_list_lw": regroup_by_group_sizes(time_list_lw, group_sizes),
        "time_list_mid": regroup_by_group_sizes(time_list_mid, group_sizes),
        "accuracy_func_list": data["accuracy_func_list"],
        "accuracy_regret_list": data["accuracy_regret_list"],
        "performance_list_up": regroup_by_group_sizes(performance_list_up, group_sizes),
        "performance_list_lw": regroup_by_group_sizes(performance_list_lw, group_sizes),
        "performance_list_mid": regroup_by_group_sizes(performance_list_mid, group_sizes),
        "phase_inner_loop_count": regroup_by_group_sizes(phase_inner_loop_count, group_sizes),
        "group_sizes": group_sizes,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_result, f, ensure_ascii=False, indent=4)


def _ratio_worker_false(sample):
    return _compute_ratio_single(sample, use_true_partitions=False)


def _ratio_worker_true(sample):
    return _compute_ratio_single(sample, use_true_partitions=True)


def _regret_worker_false(sample):
    return _compute_regret_single(sample, use_true_partitions=False)


def _regret_worker_true(sample):
    return _compute_regret_single(sample, use_true_partitions=True)


def main():
    results_dir = Path(__file__).resolve().parent / "result"
    results_dir.mkdir(parents=True, exist_ok=True)

    num_workers = _resolve_num_workers()

    with open(results_dir / "generate_decrease_K_multi_ratio.json", "r", encoding="utf-8") as f:
        ratio_data = json.load(f)
    ratio_data, ratio_group_sizes = expand_seed_runs_with_group_sizes(ratio_data)

    _run_ratio_experiment(
        data=ratio_data,
        group_sizes=ratio_group_sizes,
        use_true_partitions=False,
        output_path=results_dir / "result_decrease_K_multi_ratio.json",
        num_workers=num_workers,
    )

    with open(results_dir / "generate_decrease_K_multi_regret.json", "r", encoding="utf-8") as f:
        regret_data = json.load(f)
    regret_data, regret_group_sizes = expand_seed_runs_with_group_sizes(regret_data)

    _run_regret_experiment(
        data=regret_data,
        group_sizes=regret_group_sizes,
        use_true_partitions=False,
        output_path=results_dir / "result_decrease_K_multi_regret.json",
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()
