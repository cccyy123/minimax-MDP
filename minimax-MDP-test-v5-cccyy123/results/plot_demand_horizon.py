import json
from pathlib import Path

import matplotlib.pyplot as plt

T_list = [50+10*(i) for i in range(40)]
BASE_DIR = Path(__file__).resolve().parent
PICTURE_DIR = BASE_DIR / "picture"

# ===== Global Config =====
FIG_SIZE = (10, 6)
LINE_WIDTH = 2.2
SHOW_MARKER = False
MARKER_STYLE = "o"
MARKER_SIZE = 3
MIN_HORIZON = 70  # Set to None to disable lower-bound filtering
MAX_HORIZON = 350  # Set to None to disable horizon filtering

SHOW_DIS_LINES = 0  # 1: show dis_1/dis_2, 0: hide
SHOW_CONFIDENCE_RANGE = 1  # 1: show seed percentile band, 0: hide
CONFIDENCE_ALPHA = 0.18
SEED_BAND_LOWER_PERCENT = 5.0
SEED_BAND_UPPER_PERCENT = 95.0

X_LABEL = "Time horizon (T)"
RATIO_Y_LABEL = "Competitive ratio"
REGRET_Y_LABEL = "Regret"

X_LABEL_SIZE = 16
Y_LABEL_SIZE = 16
X_TICK_SIZE = 16
Y_TICK_SIZE = 16
LEGEND_SIZE = 25
TITLE_SIZE = 19

RATIO_TITLE = "Time horizon vs Competitive ratio"
REGRET_TITLE = "Time horizon vs Regret"

LEGEND_NAME_MAP = {
    "rgt_lw": "interval-min",
    "rgt_mid": "interval-mid",
    "rgt_up": "interval-max",
    "rgt_dis_1": "dp-acc(1)",
    "rgt_dis_2": "dp-acc(4)",
}


def mean_over_seed(group_values):
    if not isinstance(group_values, list):
        return float(group_values)
    if len(group_values) == 0:
        return 0.0
    return float(sum(group_values) / len(group_values))


def std_over_seed(group_values, mean_value):
    if not isinstance(group_values, list) or len(group_values) <= 1:
        return 0.0
    var = sum((float(v) - mean_value) ** 2 for v in group_values) / len(group_values)
    return var ** 0.5


def percentile_from_sorted(sorted_values, percent_value):
    if len(sorted_values) == 0:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    p = max(0.0, min(100.0, float(percent_value))) / 100.0
    position = p * (len(sorted_values) - 1)
    left = int(position)
    right = min(left + 1, len(sorted_values) - 1)
    weight = position - left
    return float(sorted_values[left]) * (1 - weight) + float(sorted_values[right]) * weight


def build_series(result_data):
    metric_keys = [
        k
        for k in result_data.keys()
        if k.startswith("rgt_") and not k.endswith("_time")
    ]

    series = {}
    for key in metric_keys:
        if SHOW_DIS_LINES == 0 and key in {"rgt_dis_1", "rgt_dis_2", "rgt_dis"}:
            continue

        means = []
        lowers = []
        uppers = []
        for group in result_data[key]:
            if isinstance(group, list):
                seed_values = [float(v) for v in group]
            else:
                seed_values = [float(group)]

            means.append(mean_over_seed(seed_values))
            sorted_values = sorted(seed_values)
            lowers.append(percentile_from_sorted(sorted_values, SEED_BAND_LOWER_PERCENT))
            uppers.append(percentile_from_sorted(sorted_values, SEED_BAND_UPPER_PERCENT))

        series[key] = {"mean": means, "lower": lowers, "upper": uppers}

    return series


def apply_horizon_limit(x_values, series, min_horizon, max_horizon):
    if min_horizon is None and max_horizon is None:
        return list(x_values), series

    keep_indices = []
    for i, x in enumerate(x_values):
        x_float = float(x)
        if min_horizon is not None and x_float < float(min_horizon):
            continue
        if max_horizon is not None and x_float > float(max_horizon):
            continue
        keep_indices.append(i)
    filtered_x = [x_values[i] for i in keep_indices]

    filtered_series = {}
    for key, stats in series.items():
        filtered_series[key] = {
            "mean": [stats["mean"][i] for i in keep_indices],
            "lower": [stats["lower"][i] for i in keep_indices],
            "upper": [stats["upper"][i] for i in keep_indices],
        }

    return filtered_x, filtered_series


def plot_curves(x_values, series, title, y_label, output_name):
    plt.figure(figsize=FIG_SIZE)

    for name, stats in series.items():
        y_values = stats["mean"]
        lower_values = stats["lower"]
        upper_values = stats["upper"]
        display_name = LEGEND_NAME_MAP.get(name, name)

        if SHOW_MARKER:
            line, = plt.plot(
                x_values,
                y_values,
                marker=MARKER_STYLE,
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                label=display_name,
            )
        else:
            line, = plt.plot(x_values, y_values, linewidth=LINE_WIDTH, label=display_name)

        if SHOW_CONFIDENCE_RANGE == 1:
            plt.fill_between(
                x_values,
                lower_values,
                upper_values,
                color=line.get_color(),
                alpha=CONFIDENCE_ALPHA,
                linewidth=0,
            )

    plt.xlabel(X_LABEL, fontsize=X_LABEL_SIZE)
    plt.ylabel(y_label, fontsize=Y_LABEL_SIZE)
    plt.title(title, fontsize=TITLE_SIZE)
    plt.xticks(fontsize=X_TICK_SIZE)
    plt.yticks(fontsize=Y_TICK_SIZE)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()

    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PICTURE_DIR / f"{output_name}.png", dpi=300)
    plt.close()


def main():
    with open(BASE_DIR / "generate_demand_horizon.json", "r", encoding="utf-8") as f:
        generate_demand = json.load(f)
    with open(BASE_DIR / "result_ratio_demand_horizon.json", "r", encoding="utf-8") as f:
        ratio_data = json.load(f)
    with open(BASE_DIR / "result_rgt_demand_horizon.json", "r", encoding="utf-8") as f:
        regret_data = json.load(f)

    x_values_all = generate_demand["T"]

    ratio_series = build_series(ratio_data)
    regret_series = build_series(regret_data)

    x_values, ratio_series = apply_horizon_limit(
        x_values_all,
        ratio_series,
        MIN_HORIZON,
        MAX_HORIZON,
    )
    _, regret_series = apply_horizon_limit(
        x_values_all,
        regret_series,
        MIN_HORIZON,
        MAX_HORIZON,
    )

    plot_curves(
        x_values=x_values,
        series=ratio_series,
        title=RATIO_TITLE,
        y_label=RATIO_Y_LABEL,
        output_name="result_ratio_demand_horizon",
    )

    plot_curves(
        x_values=x_values,
        series=regret_series,
        title=REGRET_TITLE,
        y_label=REGRET_Y_LABEL,
        output_name="result_rgt_demand_horizon",
    )


if __name__ == "__main__":
    main()
