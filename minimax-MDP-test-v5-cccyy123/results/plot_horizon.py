import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
PICTURE_DIR = BASE_DIR / "picture"

# ===== Plot Config (you can edit these) =====
FIG_SIZE = (10, 6)
LINE_WIDTH = 2.0
SHOW_MARKER = False
MARKER_STYLE = "o"
MARKER_SIZE = 4
MIN_HORIZON = 70  # Set to None to disable lower-bound filtering
MAX_HORIZON = 350  # Set to None to disable horizon filtering

RATIO_TIME_TITLE = "Time horizon vs Running time (under competitive ratio metric)"
REGRET_TIME_TITLE = "Time horizon vs Running time (under regret metric)"
X_LABEL = "Time horizon (T)"
TIME_Y_LABEL = "Running time (seconds)"

X_LABEL_SIZE = 16
Y_LABEL_SIZE = 16
X_TICK_SIZE = 16
Y_TICK_SIZE = 16
LEGEND_SIZE = 15
TITLE_SIZE = 17
LEGEND_LOC = "center right"
LEGEND_BBOX_TO_ANCHOR = (0.98, 0.55)

LEGEND_NAME_MAP = {
    "lw_time": "interval-min",
    "mid_time": "interval-mid",
    "up_time": "interval-max",
    "dis_1_time": "dp-acc(1)",
    "dis_2_time": "dp-acc(4)",
    "naive_greedy_time": "naive-greedy",
    "lp_time": "LP-based",
}

TIME_KEY_ORDER = [
    "lw_time",
    "mid_time",
    "up_time",
    "dis_1_time",
    "dis_2_time",
    "naive_greedy_time",
    "lp_time",
]


def mean_over_seed(group_values):
    if not isinstance(group_values, list):
        return float(group_values)
    if len(group_values) == 0:
        return 0.0
    return float(sum(group_values) / len(group_values))


def build_time_series(result_data):
    detected_keys = [k for k in result_data.keys() if k.endswith("_time")]
    time_keys = [k for k in TIME_KEY_ORDER if k in detected_keys]

    series = []
    for time_key in time_keys:
        y_values = [mean_over_seed(group) for group in result_data[time_key]]
        series.append((time_key, y_values))

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
    filtered_series = []
    for series_name, y_values in series:
        filtered_y = [y_values[i] for i in keep_indices]
        filtered_series.append((series_name, filtered_y))

    return filtered_x, filtered_series


def plot_horizon_vs_time(x_values, series, title, output_name):
    plt.figure(figsize=FIG_SIZE)
    for series_name, y_values in series:
        label = LEGEND_NAME_MAP.get(series_name, series_name)
        if SHOW_MARKER:
            plt.plot(
                x_values,
                y_values,
                linewidth=LINE_WIDTH,
                marker=MARKER_STYLE,
                markersize=MARKER_SIZE,
                label=label,
            )
        else:
            plt.plot(x_values, y_values, linewidth=LINE_WIDTH, label=label)

    plt.xlabel(X_LABEL, fontsize=X_LABEL_SIZE)
    plt.ylabel(TIME_Y_LABEL, fontsize=Y_LABEL_SIZE)
    plt.title(title, fontsize=TITLE_SIZE)
    plt.yscale("log")
    plt.xticks(fontsize=X_TICK_SIZE)
    plt.yticks(fontsize=Y_TICK_SIZE)
    plt.grid(alpha=0.25)
    plt.legend(
        fontsize=LEGEND_SIZE,
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
    )
    plt.tight_layout()

    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PICTURE_DIR / f"{output_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    with open(BASE_DIR / "generate_horizon.json", "r", encoding="utf-8") as f:
        generate_data = json.load(f)
    with open(BASE_DIR / "result_ratio_horizon.json", "r", encoding="utf-8") as f:
        ratio_data = json.load(f)
    with open(BASE_DIR / "result_rgt_horizon.json", "r", encoding="utf-8") as f:
        regret_data = json.load(f)

    x_values_all = generate_data["T"]
    ratio_time_series = build_time_series(ratio_data)
    regret_time_series = build_time_series(regret_data)

    x_values, ratio_time_series = apply_horizon_limit(
        x_values_all,
        ratio_time_series,
        MIN_HORIZON,
        MAX_HORIZON,
    )
    _, regret_time_series = apply_horizon_limit(
        x_values_all,
        regret_time_series,
        MIN_HORIZON,
        MAX_HORIZON,
    )

    plot_horizon_vs_time(
        x_values=x_values,
        series=ratio_time_series,
        title=RATIO_TIME_TITLE,
        output_name="result_ratio_horizon_time",
    )

    plot_horizon_vs_time(
        x_values=x_values,
        series=regret_time_series,
        title=REGRET_TIME_TITLE,
        output_name="result_rgt_horizon_time",
    )


if __name__ == "__main__":
    main()
