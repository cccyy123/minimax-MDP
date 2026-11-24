import argparse
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "results" / "generate_table.json"
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "result_totaldemand_interval_prob_table.json"


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
        "Input data does not match seed_repeat shape: "
        f"type={type(group_value).__name__}, "
        f"len={len(group_value) if isinstance(group_value, list) else 'NA'}, "
        f"seed_repeat={expected_repeat}"
    )


def estimate_total_demand_interval_prob(dataset):
    seed_repeat = dataset.get("seed_repeat", 1)
    seed_values = dataset.get("seed_values", list(range(seed_repeat)))

    group_results = []
    overall_full_covered_count = 0
    overall_sequence_count = 0

    grouped_iter = zip(
        dataset["L"],
        dataset["R"],
        dataset["totaldemand"],
        dataset.get("T", []),
        dataset.get("c", []),
        dataset.get("p", []),
        dataset.get("rangedemand", []),
    )

    for group_idx, (
        L_group,
        R_group,
        demand_group,
        T_value,
        c_value,
        p_value,
        rangedemand_value,
    ) in enumerate(grouped_iter):
        L_repeats = to_repeat_list(L_group, seed_repeat)
        R_repeats = to_repeat_list(R_group, seed_repeat)
        demand_repeats = to_repeat_list(demand_group, seed_repeat)

        if len(L_repeats) != len(R_repeats) or len(L_repeats) != len(demand_repeats):
            raise ValueError(f"Inconsistent repeat counts in group {group_idx}")

        if len(L_repeats) == 0:
            group_results.append(
                {
                    "group_index": group_idx,
                    "T": int(T_value),
                    "c": c_value,
                    "p": p_value,
                    "rangedemand": rangedemand_value,
                    "full_covered_sequence_count": 0,
                    "sequence_count": 0,
                    "full_covered_sequence_ratio": 0.0,
                    "sequence_full_covered_by_seed": [],
                }
            )
            continue

        sequence_flags = []
        group_full_covered_count = 0

        for seed_idx, (L_seed, R_seed, demand_seed) in enumerate(
            zip(L_repeats, R_repeats, demand_repeats)
        ):
            horizon = min(len(L_seed), len(R_seed))
            if len(L_seed) < horizon or len(R_seed) < horizon:
                raise ValueError(
                    f"L/R length mismatch in group {group_idx}: "
                    f"len(L)={len(L_seed)}, len(R)={len(R_seed)}, horizon={horizon}"
                )

            demand_value = float(demand_seed)
            all_covered = True
            for t in range(horizon):
                if not (float(L_seed[t]) <= demand_value <= float(R_seed[t])):
                    all_covered = False
                    break

            seed_label = seed_values[seed_idx] if seed_idx < len(seed_values) else seed_idx
            sequence_flags.append(
                {
                    "seed_index": seed_idx,
                    "seed_value": seed_label,
                    "is_full_covered": all_covered,
                }
            )
            if all_covered:
                group_full_covered_count += 1

        sequence_count = len(sequence_flags)
        full_covered_ratio = (
            group_full_covered_count / sequence_count if sequence_count > 0 else 0.0
        )
        overall_full_covered_count += group_full_covered_count
        overall_sequence_count += sequence_count

        group_results.append(
            {
                "group_index": group_idx,
                "T": int(T_value),
                "c": c_value,
                "p": p_value,
                "rangedemand": rangedemand_value,
                "full_covered_sequence_count": group_full_covered_count,
                "sequence_count": sequence_count,
                "full_covered_sequence_ratio": full_covered_ratio,
                "sequence_full_covered_by_seed": sequence_flags,
            }
        )

    return {
        "seed_repeat": seed_repeat,
        "group_results": group_results,
        "overall_full_covered_sequence_count": overall_full_covered_count,
        "overall_sequence_count": overall_sequence_count,
        "overall_full_covered_sequence_ratio": (
            overall_full_covered_count / overall_sequence_count
            if overall_sequence_count > 0
            else 0.0
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate sequence-level full coverage for each seed, where full coverage "
            "means total demand is within [L, R] for every time step."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to generate_*.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    result = estimate_total_demand_interval_prob(dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(result["overall_full_covered_sequence_count"])
    print(result["overall_full_covered_sequence_ratio"])
    print(f"Saved probability result to: {args.output}")


if __name__ == "__main__":
    main()