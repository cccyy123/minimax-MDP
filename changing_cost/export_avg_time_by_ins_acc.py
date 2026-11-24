import argparse
import json
import math
from pathlib import Path
from typing import Any, List


# Global defaults:
# - DEFAULT_PERFORMANCE: which performance result file to read (regret or ratio)
# - DEFAULT_METHOD: which time list to use (up, mid, or lw)
DEFAULT_PERFORMANCE = "regret"
DEFAULT_METHOD = "up"

# Default target accuracy_func. CLI can still override this value.
DEFAULT_ACCURACY_FUNC = 6.0


def _is_close(a: float, b: float, tol: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def _mean_seed_time(value: Any) -> float:
    # Typical case: one config -> list of per-seed running times.
    if isinstance(value, list):
        if not value:
            return float("nan")
        return float(sum(value) / len(value))
    # Defensive fallback: allow scalar values.
    return float(value)


def _infer_group_size(accuracy_func_list: List[float], tol: float) -> int:
    if not accuracy_func_list:
        raise ValueError("accuracy_func_list is empty")

    first = float(accuracy_func_list[0])
    for i in range(1, len(accuracy_func_list)):
        if _is_close(float(accuracy_func_list[i]), first, tol):
            return i
    return len(accuracy_func_list)


def _build_group_labels(num_groups: int, ins_acc_values: List[float]) -> List[str]:
    labels = [f"{x:g}" for x in ins_acc_values]
    if num_groups == len(labels):
        return labels
    if num_groups == len(labels) + 1:
        return labels + ["oracle"]
    return [f"group_{i}" for i in range(num_groups)]


def _select_target_accuracy(
    accuracy_values: List[float],
    requested: float,
    tol: float,
) -> tuple[float, List[int], bool]:
    """
    Return selected target, matched indices, and whether match is exact.

    Rule:
    1) If any value equals requested within tol, use those exact matches.
    2) Otherwise use the nearest available accuracy value.
    """
    exact_indices = [
        i for i, a in enumerate(accuracy_values) if _is_close(a, requested, tol)
    ]
    if exact_indices:
        return requested, exact_indices, True

    unique_values = sorted(set(accuracy_values))
    nearest = min(unique_values, key=lambda x: abs(x - requested))
    nearest_indices = [
        i for i, a in enumerate(accuracy_values) if _is_close(a, nearest, tol)
    ]
    return nearest, nearest_indices, False


def _normalize_accuracy_list(raw_accuracy: List[Any], target_len: int, tol: float) -> List[float]:
    """
    Align accuracy_func_list to config-level length.

    Some result files expand accuracy_func_list per seed (e.g., len = target_len * seed_repeat),
    while time_list is already config-level (len = target_len).
    """
    if len(raw_accuracy) == target_len:
        return [float(x) for x in raw_accuracy]

    if target_len <= 0 or len(raw_accuracy) % target_len != 0:
        raise ValueError(
            f"Cannot normalize accuracy_func_list length {len(raw_accuracy)} to target length {target_len}"
        )

    repeat = len(raw_accuracy) // target_len
    compressed: List[float] = []
    for i in range(target_len):
        block = raw_accuracy[i * repeat : (i + 1) * repeat]
        first = float(block[0])
        if any(not _is_close(float(x), first, tol) for x in block[1:]):
            raise ValueError(
                "accuracy_func_list is not constant within repeated seed blocks; cannot normalize safely"
            )
        compressed.append(first)
    return compressed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export average running time by ins_acc at a fixed accuracy_func. "
            "Average is taken over seeds."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["regret", "ratio"],
        default=DEFAULT_PERFORMANCE,
        help="Which result file to use.",
    )
    parser.add_argument(
        "--method",
        choices=["up", "mid", "lw"],
        default=DEFAULT_METHOD,
        help="Which time list to use: time_list_up/time_list_mid/time_list_lw.",
    )
    parser.add_argument(
        "--accuracy-func",
        type=float,
        default=DEFAULT_ACCURACY_FUNC,
        help=(
            "Target accuracy_func. If no exact value exists, the nearest available "
            "accuracy_func will be used."
        ),
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for matching accuracy_func.",
    )
    parser.add_argument(
        "--ins-acc",
        type=str,
        default="1.04,1.07,1.1",
        help="Comma-separated ins_acc values for group labels.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output json path. Default: result/avg_time_by_ins_acc_<dataset>_<method>_acc_<accuracy>.json",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    result_dir = base_dir / "result"
    source_path = result_dir / f"result_decrease_K_multi_{args.dataset}.json"

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    accuracy_func_list = data.get("accuracy_func_list")
    time_key = f"time_list_{args.method}"
    time_list = data.get(time_key)
    loop_count_key = "phase_inner_loop_count"
    loop_count_list = data.get(loop_count_key)
    if loop_count_list is None:
        # Backward compatibility with older intermediate key design.
        loop_count_key = f"phase_inner_loop_count_{args.method}"
        loop_count_list = data.get(loop_count_key)

    if accuracy_func_list is None:
        raise KeyError("Missing key: accuracy_func_list")
    if time_list is None:
        raise KeyError(f"Missing key: {time_key}")
    accuracy_func_list = _normalize_accuracy_list(
        raw_accuracy=accuracy_func_list,
        target_len=len(time_list),
        tol=args.tol,
    )

    ins_acc_values = [float(x.strip()) for x in args.ins_acc.split(",") if x.strip()]

    group_size = _infer_group_size(accuracy_func_list, args.tol)
    if group_size <= 0:
        raise ValueError("Inferred invalid group size")
    if len(accuracy_func_list) % group_size != 0:
        raise ValueError("Cannot infer a clean group split from accuracy_func_list")

    num_groups = len(accuracy_func_list) // group_size
    group_labels = _build_group_labels(num_groups, ins_acc_values)

    selected_accuracy, matched_indices, is_exact_match = _select_target_accuracy(
        accuracy_values=accuracy_func_list,
        requested=args.accuracy_func,
        tol=args.tol,
    )

    grouped_times: List[List[float]] = [[] for _ in range(num_groups)]
    grouped_loops: List[List[float]] = [[] for _ in range(num_groups)]
    for idx in matched_indices:
        g = idx // group_size
        if g < 0 or g >= num_groups:
            continue
        grouped_times[g].append(_mean_seed_time(time_list[idx]))
        if loop_count_list is not None:
            grouped_loops[g].append(_mean_seed_time(loop_count_list[idx]))

    results = []
    for g, label in enumerate(group_labels):
        time_values = grouped_times[g]
        avg_time = float(sum(time_values) / len(time_values)) if time_values else float("nan")
        loop_values = grouped_loops[g]
        avg_loop = float(sum(loop_values) / len(loop_values)) if loop_values else float("nan")
        result_item = {
            "group_index": g,
            "ins_acc": label,
            "avg_running_time": avg_time,
            "num_configs_matched": len(time_values),
        }
        if loop_count_list is not None:
            result_item["avg_phase_inner_loop_count"] = avg_loop
        results.append(result_item)

    output = args.output.strip()
    if not output:
        acc_token = str(args.accuracy_func).replace(".", "p")
        output_path = result_dir / (
            f"avg_time_by_ins_acc_{args.dataset}_{args.method}_acc_{acc_token}.json"
        )
    else:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = base_dir / output_path

    payload = {
        "source_file": str(source_path),
        "dataset": args.dataset,
        "method": args.method,
        "accuracy_func_requested": args.accuracy_func,
        "accuracy_func_selected": selected_accuracy,
        "is_exact_match": is_exact_match,
        "tolerance": args.tol,
        "group_size": group_size,
        "num_groups": num_groups,
        "matched_indices": matched_indices,
        "loop_count_key": loop_count_key if loop_count_list is not None else None,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
