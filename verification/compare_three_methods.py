#!/usr/bin/env python3
"""Closed forms versus the EC.3.4 single-switch LP and EC.7.4 backward DP.

The single-switch ratio is a restricted-adversary optimum, not a guarantee
against arbitrary predictions and not Feng et al.'s imbalance-cost ratio.
DP optimizes each metric separately; interval and order grids use spacing h.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import sys
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from math import gcd, lcm
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Instance:
    name: str
    lower: Q
    upper: Q
    deltas: tuple[Q, ...]
    capacities: tuple[Q, ...]
    p: Q
    c: Q

    @property
    def T(self):
        return len(self.deltas)

    def as_dict(self):
        return {"name": self.name, "lower": str(self.lower), "upper": str(self.upper),
                "deltas": list(map(str, self.deltas)), "capacities": list(map(str, self.capacities)),
                "p": str(self.p), "c": str(self.c)}


def instance(name, lower, upper, deltas, capacities, p=2, c=1):
    a = Instance(name, Q(str(lower)), Q(str(upper)), tuple(Q(str(x)) for x in deltas),
                 tuple(Q(str(x)) for x in capacities), Q(str(p)), Q(str(c)))
    if not 0 < a.c < a.p or not 0 < a.lower <= a.upper:
        raise ValueError("Require 0<c<p and 0<lower<=upper.")
    if not a.T or len(a.capacities) != a.T or any(v <= 0 for v in a.capacities):
        raise ValueError("Require equal positive lengths and positive capacities.")
    if sum(a.capacities) < a.upper:
        raise ValueError("Require total capacity >= demand upper bound.")
    previous = a.upper-a.lower
    for delta in a.deltas:
        if not 0 <= delta <= previous:
            raise ValueError("Prediction widths must be nonnegative and nonincreasing.")
        previous = delta
    return a


@dataclass(frozen=True)
class Node:
    parent: int
    depth: int
    lo: int
    hi: int


def closed_forms(ins):
    tail, gamma, phi = Q(0), [], []
    for delta, capacity in zip(reversed(ins.deltas), reversed(ins.capacities)):
        gamma.append(ins.c*(ins.p-ins.c)/ins.p*(delta-tail))
        phi.append((ins.p*ins.lower+ins.c*tail)/(ins.p*ins.lower+ins.c*delta))
        tail += capacity
    return {"regret": str(max(gamma)), "ratio": str(min(phi))}


def examples():
    return [
        (instance("single_period", 1, 3, [2], [3], 3, 1),
         "Single period; fractional optimal orders; no prediction updates."),
        (instance("EC_3_4", 1, 7, [5, 4, 1], [4, 2, 1], 2, 1),
         "Existing EC.3.4 counterexample; declining capacity and an abrupt final accuracy improvement."),
        (instance("four_period", 2, 8, [5, 4, 3, 1], [4, 2, 1, 1], 3, 1),
         "Four periods; higher demand floor, declining then flat capacity, gradual accuracy gains followed by a final jump."),
        (instance("six_period", 2, 8, [5, 5, 4, 3, 2, 1], [3, 2, 1, "0.5", 1, "0.5"], 3, 2),
         "Six periods; an initial accuracy plateau, a capacity rebound on day 5, and a smaller profit margin."),
        (instance("ten_period", "0.5", "4.5", ["3.5", "3.5", 3, 3, "2.5", 2, "1.5", 1, "0.5", "0.5"],
                  ["1.5", "0.5", "0.5", "0.5", "0.3", "0.2", "0.2", "0.1", "0.1", "0.6"], 3, 1),
         "Ten periods; multiple accuracy plateaus, fractional demand/capacity, and a final capacity rebound."),
    ]


def rational_gcd(values):
    denominator = lcm(*(v.denominator for v in values))
    numerator = gcd(*(int(v*denominator) for v in values))
    return Q(numerator, denominator)


def single_switch_tree(ins: Instance):
    """Exactly the T+1 sequences defined in manuscript EC.3.4.

    Prefixes, not just equal current intervals, share decisions. All choices
    after the switch are optimized too; no emulator for other paths is claimed.
    h is only an exact integer encoding of these rational endpoints, not a mesh.
    """
    h = rational_gcd((ins.upper-ins.lower,)+ins.deltas)
    if not h:
        h = Q(1)
    nodes = [Node(-1, 0, 0, int((ins.upper-ins.lower)/h))]
    prefixes = {}
    for switch in range(ins.T+1):
        lower, upper, parent = ins.lower, ins.upper, 0
        for t, delta in enumerate(ins.deltas, 1):
            if t <= switch:
                lower = upper-delta
            else:
                upper = lower+delta
            lo, hi = int((lower-ins.lower)/h), int((upper-ins.lower)/h)
            key = (parent, lo, hi)
            if key not in prefixes:
                prefixes[key] = len(nodes)
                nodes.append(Node(parent, t, lo, hi))
            parent = prefixes[key]
    return h, nodes


def single_switch_lp(ins, metric):
    h, nodes = single_switch_tree(ins)
    if metric not in ("regret", "ratio"):
        raise ValueError(metric)
    margin = ins.p-ins.c
    rows = []
    for j, node in enumerate(nodes[1:], 1):
        up, down = [(j, Q(1))], [(j, Q(-1))]
        if node.parent:
            up.append((node.parent, Q(-1)))
            down.append((node.parent, Q(1)))
        rows.extend([(up, ins.capacities[node.depth-1]), (down, Q(0))])
        if node.depth == ins.T:
            for tick in sorted({node.lo, node.hi}):
                d = ins.lower+h*tick
                if metric == "regret":
                    rows.extend([([(j, -margin), (0, Q(-1))], -margin*d),
                                 ([(j, ins.c), (0, Q(-1))], ins.c*d)])
                else:
                    rows.extend([([(j, -margin), (0, margin*d)], Q(0)),
                                 ([(j, ins.c), (0, margin*d)], ins.p*d)])
    rr, cc, vv = [], [], []
    for i, (row, _) in enumerate(rows):
        for j, v in row:
            rr.append(i)
            cc.append(j)
            vv.append(float(v))
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(rows), len(nodes))).tocsr()
    rhs = np.array([float(b) for _, b in rows])
    upper = [margin*ins.upper if metric == "regret" else Q(1)]
    upper += [sum(ins.capacities[:n.depth]) for n in nodes[1:]]
    objective = np.zeros(len(nodes))
    objective[0] = 1 if metric == "regret" else -1
    start = time.perf_counter()
    solution = linprog(objective, A_ub=matrix, b_ub=rhs, bounds=[(0, float(v)) for v in upper],
                       method="highs-ds", options={"primal_feasibility_tolerance": 1e-9,
                       "dual_feasibility_tolerance": 1e-9, "time_limit": 120})
    seconds = time.perf_counter()-start
    if not solution.success:
        raise RuntimeError(solution.message)
    # Exact rational certificate for this restricted LP. Any y<=0 provides
    # b'y + sum_j min(0, (f-A'y)_j*u_j) as a minimization lower bound.
    reduced = [Q(int(objective[j])) for j in range(len(nodes))]
    dual = Q(0)
    for (row, b), multiplier in zip(rows, solution.ineqlin.marginals):
        y = min(Q(0), Q.from_float(float(multiplier)))
        dual += b*y
        for j, v in row:
            reduced[j] -= v*y
    dual += sum(min(Q(0), r*u) for r, u in zip(reduced, upper))
    raw = [Q(0)]+[Q.from_float(float(x)) for x in solution.x[1:]]
    inventory = [Q(0)]*len(nodes)
    worst = Q(0) if metric == "regret" else Q(1)
    for j, node in enumerate(nodes[1:], 1):
        inventory[j] = inventory[node.parent]+min(ins.capacities[node.depth-1],
                           max(Q(0), raw[j]-raw[node.parent]))
        if node.depth == ins.T:
            for tick in {node.lo, node.hi}:
                d = ins.lower+h*tick
                profit = ins.p*min(d, inventory[j])-ins.c*inventory[j]
                payoff = margin*d-profit if metric == "regret" else profit/(margin*d)
                worst = max(worst, payoff) if metric == "regret" else min(worst, payoff)
    lo, hi = (max(Q(0), dual), min(upper[0], worst)) if metric == "regret" else (max(Q(0), worst), min(Q(1), -dual))
    if not 0 <= hi-lo <= Q("1e-7"):
        raise RuntimeError("LP rational certificate failed.")
    violation = max(0., float(np.max(matrix@solution.x-rhs)), float(np.max(-solution.x)),
                    float(np.max(solution.x-np.array(list(map(float, upper))))))
    return {"value": float(solution.x[0]), "lower": str(lo), "upper": str(hi),
            "certificate_gap": str(hi-lo), "max_primal_violation": violation,
            "variables": len(nodes), "constraints": len(rows), "solve_seconds": seconds,
            "scope": "EC.3.4 single-switch restricted adversary; continuous orders"}


def forward_extreme(values, count, axis, maximize):
    """In-place extremum over entries i,...,i+count-1, clipped to valid indices."""
    function = maximum_filter1d if maximize else minimum_filter1d
    return function(values, size=count, axis=axis, origin=-(count//2),
                    mode="constant", cval=-np.inf if maximize else np.inf, output=values)


def backward_dp(ins: Instance, h: Q, metric: str, max_cells=250_000_000):
    """EC.7.4 recursion, with valid index ranges and Delta_0/Delta_{T+1}.

    Widths are maximal: narrower observations can be padded online to nested
    intervals of these widths. Terminal state is a point demand. Each stage
    enumerates every feasible grid shift/order via exact sliding extrema.
    Closed forms and single-switch LP values are not used here.
    """
    if metric not in ("regret", "ratio") or h <= 0:
        raise ValueError("Require metric regret/ratio and positive mesh.")
    ratios = ((ins.upper-ins.lower)/h,)+tuple(d/h for d in ins.deltas)+tuple(v/h for v in ins.capacities)
    if any(v.denominator != 1 for v in ratios):
        raise ValueError("Demand width, prediction widths and all capacities must be multiples of h.")
    N, M = int((ins.upper-ins.lower)/h), int(sum(ins.capacities)/h)
    if (N+1)*(M+1) > max_cells:
        raise ValueError("DP exceeds --max-cells; increase explicitly or use a coarser mesh.")
    start = time.perf_counter()
    demand = float(ins.lower)+float(h)*np.arange(N+1)[:, None]
    inventory = float(h)*np.arange(M+1)[None, :]
    # Keep one dense DP array: finer grids otherwise create several large
    # temporary arrays. Sliding filters also overwrite this array in place.
    A = np.minimum(demand, inventory)
    A *= float(ins.p)
    A -= float(ins.c)*inventory
    hindsight = float(ins.p-ins.c)*demand
    if metric == "regret":
        np.subtract(hindsight, A, out=A)
    else:
        np.divide(A, hindsight, out=A)
    widths = [N]+[int(d/h) for d in ins.deltas]+[0]
    state_cells = A.size
    for t in range(ins.T, -1, -1):
        # B(t): adversary reveals the next interval (or terminal demand).
        shifts = widths[t]-widths[t+1]
        B = forward_extreme(A, shifts+1, axis=0, maximize=metric == "regret")
        B = B[:N-widths[t]+1]
        if t == 0:
            value = float(B[0, 0])
            break
        # A(t): choose the current order after observing the current interval.
        A = forward_extreme(B, int(ins.capacities[t-1]/h)+1, axis=1,
                            maximize=metric == "ratio")
        state_cells += A.size
    # Exact real-arithmetic discretization envelopes; floating evaluation is
    # separately checked on small games against exact Fraction enumeration.
    if metric == "regret":
        below = float((ins.p-ins.c)*h)
        above = float(max(ins.c, ins.p-ins.c)*h)
        lower, upper = max(0., value-below), value+above
    else:
        lower = max(0., value)*float(ins.lower/(ins.lower+h))
        upper = min(1., value+float(h/ins.lower))
    return {"value": value, "mesh": str(h),
            "continuous_lower_discretization_only": lower,
            "continuous_upper_discretization_only": upper,
            "state_cells": state_cells, "peak_array_cells": (N+1)*(M+1),
            "seconds": time.perf_counter()-start}


def exact_small_dp(ins, h, metric):
    """Independent Fraction reference: explicitly enumerate ALL interval widths.

    Only used on tiny test games; no sliding filters or maximal-width reduction.
    """
    @lru_cache(None)
    def after_observation(t, lower, upper, inventory):
        outcomes = []
        for order in range(int(ins.capacities[t]/h)+1):
            q = inventory+order
            if t == ins.T-1:
                payoffs = []
                for d_tick in range(lower, upper+1):
                    d, stock = ins.lower+h*d_tick, h*q
                    profit = ins.p*min(d, stock)-ins.c*stock
                    payoff = (ins.p-ins.c)*d-profit if metric == "regret" else profit/((ins.p-ins.c)*d)
                    payoffs.append(payoff)
                outcomes.append(max(payoffs) if metric == "regret" else min(payoffs))
            else:
                outcomes.append(before_observation(t+1, lower, upper, q))
        return min(outcomes) if metric == "regret" else max(outcomes)

    @lru_cache(None)
    def before_observation(t, lower, upper, inventory):
        outcomes = [after_observation(t, a, b, inventory)
                    for a in range(lower, upper+1)
                    for b in range(a, min(upper, a+int(ins.deltas[t]/h))+1)]
        return max(outcomes) if metric == "regret" else min(outcomes)

    return before_observation(0, 0, int((ins.upper-ins.lower)/h), 0)


def self_test():
    comparisons = 0
    # Exact full-width enumeration checks the reduction to maximal intervals,
    # the order of play, terminal payoff, index limits and both objectives.
    for deltas in ([2], [2, 1], [1, 0], [2, 2, 1], [2, 1, 0]):
        T = len(deltas)
        for p, c in ((2, 1), (5, 2)):
            ins = instance("reference", 1, 3, deltas, [3] if T == 1 else [1]*(T-1)+[2], p, c)
            for metric in ("regret", "ratio"):
                for h in (Q(1), Q("0.5")):
                    expected = exact_small_dp(ins, h, metric)
                    actual = backward_dp(ins, h, metric)["value"]
                    if abs(actual-float(expected)) > 1e-10:
                        raise RuntimeError(f"DP check failed: {deltas}, {metric}, h={h}: {actual} != {expected}")
                    comparisons += 1
    one = examples()[0][0]
    for metric, value in (("regret", Q(4, 3)), ("ratio", Q(3, 5))):
        r = single_switch_lp(one, metric)
        if not Q(r["lower"]) <= value <= Q(r["upper"]):
            raise RuntimeError("Analytical one-period LP check failed.")
    ec = examples()[1][0]
    r = single_switch_lp(ec, "ratio")
    if not Q(r["lower"]) <= Q(2, 3) <= Q(r["upper"]):
        raise RuntimeError("EC.3.4 restricted ratio check failed.")
    if abs(backward_dp(ec, Q("0.1"), "ratio")["value"]-0.5) > 1e-10:
        raise RuntimeError("EC.3.4 full-game ratio check failed.")
    # Explicitly catch the floor(k*delta) error in the printed recurrence:
    # equal consecutive widths must allow only zero shift even when delta<1.
    fixed = instance("no_update", 1, 3, [2, 2], [1, 2])
    if abs(backward_dp(fixed, Q("0.5"), "regret")["value"]-1) > 1e-10:
        raise RuntimeError("Zero-shift transition check failed.")
    try:
        backward_dp(one, Q("0.7"), "regret")
    except ValueError:
        pass
    else:
        raise RuntimeError("Nonaligned mesh was not rejected.")
    print(f"Self-tests passed: {comparisons} exact full-interval DP comparisons, analytical LP checks, EC.3.4 and guards.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", nargs="+", default=["0.005", "0.001", "0.0005"])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"three_method_results")
    parser.add_argument("--max-cells", type=int, default=250_000_000)
    parser.add_argument("--self-test", action="store_true", help="Run exact small-game checks before the experiments.")
    args = parser.parse_args()
    meshes = sorted(set(map(Q, args.meshes)), reverse=True)
    if not meshes or min(meshes) <= 0:
        parser.error("Meshes must be positive.")
    if args.self_test:
        self_test()
    records = []
    for ins, description in examples():
        record = {"instance": ins.as_dict(), "description": description, "metrics": {}}
        for metric in ("regret", "ratio"):
            lp = single_switch_lp(ins, metric)
            dp = [backward_dp(ins, h, metric, args.max_cells) for h in meshes]
            exact = closed_forms(ins)[metric]
            cf = float(Q(exact))
            if metric == "regret" and abs(lp["value"]-cf) > 1e-7:
                raise RuntimeError(f"{ins.name}: single-switch regret disagrees with the closed form.")
            if metric == "ratio" and lp["value"] < cf-1e-7:
                raise RuntimeError(f"{ins.name}: restricted-adversary ratio is below the closed form.")
            for d in dp:
                if not (d["continuous_lower_discretization_only"]-1e-10 <= cf <=
                        d["continuous_upper_discretization_only"]+1e-10):
                    raise RuntimeError(f"{ins.name}: DP discretization bounds exclude the closed form.")
            record["metrics"][metric] = {"closed_form": exact, "single_switch_lp": lp, "dp": dp}
            print(f"{ins.name:15s} {metric:6s}: closed={exact}, SS-LP={lp['value']:.10f}, "
                  +", ".join(f"DP({r['mesh']})={r['value']:.10f}" for r in dp), flush=True)
        records.append(record)
    gap_count = sum(r["metrics"]["ratio"]["single_switch_lp"]["value"]-
                    float(Q(r["metrics"]["ratio"]["closed_form"])) > 1e-7 for r in records)
    if len(records) != 5 or len({len(r["instance"]["deltas"]) for r in records}) != 5 or gap_count < 4:
        raise RuntimeError("The requested five distinct horizons / four ratio gaps are not satisfied.")
    metadata = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "command": sys.argv, "self_tests_run": args.self_test,
                "lp_primal_dual_feasibility_tolerance": 1e-9,
                "lp_solver": "HiGHS dual simplex",
                "dp_arithmetic": "float64; no iterative optimizer tolerance",
                "instances_with_single_switch_ratio_gap": gap_count,
                "selection": "One analytical control and four deliberately chosen single-switch ratio counterexamples; not a random sample.",
                "warning": "Single-switch ratio is a restricted-game optimum, not a full-game performance guarantee."}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"results.json").write_text(json.dumps({"metadata": metadata, "results": records}, indent=2)+"\n")
    lines = ["# 三种方法的最优值比较", "",
             "LP 为 EC.3.4 的 single-switch 限制博弈；DP 为 EC.7.4 的反向递推，每个指标分别优化。",
             "single-switch ratio 列不是完整博弈保证，也不是 Feng 等人的 imbalance-cost ratio。", "",
             "五个实例包括一个单期对照和四个特意选取的 single-switch ratio 存在差异的实例，不是随机抽样。", "",
             "| 实例 | T | 初始区间 | Δ | V | (p,c) |", "|---|---:|---|---|---|---|"]
    for r in records:
        a = r["instance"]
        lines.append(f"| {a['name']} | {len(a['deltas'])} | [{a['lower']},{a['upper']}] | {', '.join(a['deltas'])} | {', '.join(a['capacities'])} | ({a['p']},{a['c']}) |")
    lines += ["", f"结果使用最细 DP 网格 δ={min(meshes)}。", "",
              "| 实例 | Γ 闭式 | Γ single-switch LP | Γ DP | Φ 闭式 | Φ single-switch LP | Φ DP |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        cells = []
        for metric in ("regret", "ratio"):
            a = r["metrics"][metric]
            cells += [a["closed_form"], f"{a['single_switch_lp']['value']:.8f}", f"{a['dp'][-1]['value']:.8f}"]
        lines.append("| "+r['instance']['name']+" | "+" | ".join(cells)+" |")
    lines += ["", "## DP 网格敏感性", "",
              "| 实例 | 指标 | δ | DP 值 | 与闭式绝对差 | 离散化包络下界 | 离散化包络上界 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in records:
        for metric, a in r["metrics"].items():
            for d in a["dp"]:
                lines.append(f"| {r['instance']['name']} | {metric} | {d['mesh']} | {d['value']:.10f} | {abs(d['value']-float(Q(a['closed_form']))):.3g} | {d['continuous_lower_discretization_only']:.10f} | {d['continuous_upper_discretization_only']:.10f} |")
    lines += ["", "包络只反映离散化，不是浮点数值的严格区间证书；推导、DP 公式修正及方法边界见 ../README.md。", ""]
    (args.output/"comparison.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
