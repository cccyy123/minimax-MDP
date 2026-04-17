import json
from pathlib import Path

import matplotlib.pyplot as plt


METHOD_ORDER = ("up", "mid", "lw")
PLOT_TRUE_TAUK_POINTS = False
DEFAULT_METHOD_STYLES = {
    "up": {"color": "#1f77b4", "marker": "o", "label": "up"},
    "mid": {"color": "#2ca02c", "marker": "^", "label": "mid"},
    "lw": {"color": "#d62728", "marker": "s", "label": "lw"},
}


def _build_method_styles(style_cfg):
    styles = {}
    legend_labels = style_cfg.get("legend_labels", {})
    for method in METHOD_ORDER:
        default_style = DEFAULT_METHOD_STYLES[method]
        styles[method] = {
            "color": default_style["color"],
            "marker": default_style["marker"],
            "label": legend_labels.get(method, default_style["label"]),
        }
    return styles


def _apply_axes_config(ax, plot_cfg):
    xlabel = plot_cfg.get("xlabel")
    ylabel = plot_cfg.get("ylabel")
    title = plot_cfg.get("title")

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=plot_cfg.get("xlabel_fontsize"))
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=plot_cfg.get("ylabel_fontsize"))
    if title is not None:
        ax.set_title(title, fontsize=plot_cfg.get("title_fontsize"))

    ax.tick_params(
        axis="x",
        labelsize=plot_cfg.get("xtick_labelsize"),
    )
    ax.tick_params(
        axis="y",
        labelsize=plot_cfg.get("ytick_labelsize"),
    )

    xticks = plot_cfg.get("xticks")
    if xticks is not None:
        ax.set_xticks(xticks)
    xtick_labels = plot_cfg.get("xtick_labels")
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels)

    yticks = plot_cfg.get("yticks")
    if yticks is not None:
        ax.set_yticks(yticks)
    ytick_labels = plot_cfg.get("ytick_labels")
    if ytick_labels is not None:
        ax.set_yticklabels(ytick_labels)

    legend_title = plot_cfg.get("legend_title")
    if plot_cfg.get("show_legend", True):
        ax.legend(
            fontsize=plot_cfg.get("legend_fontsize"),
            title=legend_title,
            title_fontsize=plot_cfg.get("legend_title_fontsize"),
        )


def _mean_per_group(nested):
    means = []
    for group in nested:
        if len(group) == 0:
            means.append(float("nan"))
        else:
            means.append(sum(group) / len(group))
    return means


def _maybe_drop_true_group(nested, result_data, include_true_group):
    if include_true_group:
        return nested

    group_sizes = result_data.get("group_sizes")
    if not group_sizes:
        return nested

    true_group_size = group_sizes[-1]
    if not isinstance(true_group_size, int):
        return nested
    if true_group_size <= 0 or true_group_size > len(nested):
        return nested
    return nested[:-true_group_size]


def _get_method_means(result_data, method, include_true_group):
    perf_nested = _maybe_drop_true_group(
        result_data[f"performance_list_{method}"],
        result_data,
        include_true_group,
    )
    time_nested = _maybe_drop_true_group(
        result_data[f"time_list_{method}"],
        result_data,
        include_true_group,
    )
    return _mean_per_group(perf_nested), _mean_per_group(time_nested)


def _collect_points(result_data, include_true_group):
    points = []
    for method in METHOD_ORDER:
        perf, t = _get_method_means(result_data, method, include_true_group)
        for p_val, t_val in zip(perf, t):
            points.append({"method": method, "perf": p_val, "time": t_val})
    return points


def _is_dominated(point_i, all_points, perf_mode):
    for point_j in all_points:
        if point_j is point_i:
            continue

        if perf_mode == "max":
            better_perf = point_j["perf"] > point_i["perf"]
        else:
            better_perf = point_j["perf"] < point_i["perf"]

        better_time = point_j["time"] < point_i["time"]
        if better_perf and better_time:
            return True
    return False


def _filter_non_dominated(points, perf_mode):
    kept = []
    for method in METHOD_ORDER:
        method_points = [p for p in points if p["method"] == method]
        kept.extend([p for p in method_points if not _is_dominated(p, method_points, perf_mode)])
    return kept


def _plot_one(result_data, output_path, plot_cfg, include_true_group):
    method_styles = _build_method_styles(plot_cfg)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for method, style in method_styles.items():
        perf, t = _get_method_means(result_data, method, include_true_group)
        ax.scatter(
            perf,
            t,
            c=style["color"],
            label=style["label"],
            marker=style["marker"],
            s=34,
            alpha=0.85,
            edgecolors="none",
        )

    if plot_cfg.get("y_log", True):
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.35)
    _apply_axes_config(ax, plot_cfg)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_filtered_points(points, output_path, plot_cfg):
    method_styles = _build_method_styles(plot_cfg)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    for method, style in method_styles.items():
        method_points = [p for p in points if p["method"] == method]
        perf = [p["perf"] for p in method_points]
        t = [p["time"] for p in method_points]
        ax.scatter(
            perf,
            t,
            c=style["color"],
            label=style["label"],
            marker=style["marker"],
            s=34,
            alpha=0.85,
            edgecolors="none",
        )

    if plot_cfg.get("y_log", True):
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.35)
    _apply_axes_config(ax, plot_cfg)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    base_dir = Path(__file__).resolve().parent
    result_dir = base_dir / "result"

    ratio_file = result_dir / "result_decrease_K_multi_ratio.json"
    regret_file = result_dir / "result_decrease_K_multi_regret.json"

    if not ratio_file.exists() or not regret_file.exists():
        missing = []
        if not ratio_file.exists():
            missing.append(str(ratio_file))
        if not regret_file.exists():
            missing.append(str(regret_file))
        raise FileNotFoundError("Missing result file(s): " + ", ".join(missing))

    with open(ratio_file, "r", encoding="utf-8") as f:
        ratio_data = json.load(f)
    with open(regret_file, "r", encoding="utf-8") as f:
        regret_data = json.load(f)

    ratio_out = result_dir / "scatter_ratio_time_log_vs_performance.png"
    regret_out = result_dir / "scatter_regret_time_log_vs_performance.png"
    ratio_filtered_out = result_dir / "scatter_ratio_time_log_vs_performance_filtered.png"
    regret_filtered_out = result_dir / "scatter_regret_time_log_vs_performance_filtered.png"
    include_true_group = PLOT_TRUE_TAUK_POINTS
    lengend_size=20
    # Customize each figure independently.
    # legend_labels keys: up/mid/lw
    plot_cfg = {
        "ratio": {
            "xlabel": "Competitive ratio",
            "xlabel_fontsize": 17,
            "ylabel": "Running time (seconds)",
            "ylabel_fontsize": 17,
            "title": "Competitive ratio vs Running time",
            "title_fontsize": 17,
            "xtick_labelsize": 17,
            "ytick_labelsize": 17,
            "legend_fontsize": lengend_size,
            "legend_title": None,
            "legend_title_fontsize": 17,
            "legend_labels": {"up": "interval-max", "mid": "interval-mid", "lw": "interval-min"},
            "y_log": True,
            # "xticks": [0.1, 0.2, 0.3],
            # "xtick_labels": ["A", "B", "C"],
            # "yticks": [1e-2, 1e-1, 1],
            # "ytick_labels": ["0.01", "0.1", "1"],
        },
        "regret": {
            "xlabel": "Regret",
            "xlabel_fontsize": 17,
            "ylabel": "Running time (seconds)",
            "ylabel_fontsize": 17,
            "title": "Regret vs Running time",
            "title_fontsize": 17,
            "xtick_labelsize": 17,
            "ytick_labelsize": 17,
            "legend_fontsize": lengend_size,
            "legend_title": None,
            "legend_title_fontsize": 17,
            "legend_labels": {"up": "interval-max", "mid": "interval-mid", "lw": "interval-min"},
            "y_log": True,
        },
        "ratio_filtered": {
            "xlabel": "Competitive ratio",
            "xlabel_fontsize": 17,
            "ylabel": "Running time (seconds)",
            "ylabel_fontsize": 17,
            "title": "Competitive ratio vs Running time (non-dominated only)",
            "title_fontsize": 17,
            "xtick_labelsize": 17,
            "ytick_labelsize": 17,
            "legend_fontsize": lengend_size,
            "legend_title": None,
            "legend_title_fontsize": 17,
            "legend_labels": {"up": "interval-max", "mid": "interval-mid", "lw": "interval-min"},
            "y_log": True,
        },
        "regret_filtered": {
            "xlabel": "Regret",
            "xlabel_fontsize": 17,
            "ylabel": "Running time (seconds)",
            "ylabel_fontsize": 17,
            "title": "Regret vs Running time (non-dominated only)",
            "title_fontsize": 17,
            "xtick_labelsize": 17,
            "ytick_labelsize": 17,
            "legend_fontsize": lengend_size,
            "legend_title": None,
            "legend_title_fontsize": 17,
            "legend_labels": {"up": "interval-max", "mid": "interval-mid", "lw": "interval-min"},
            "y_log": True,
        },
    }

    _plot_one(
        result_data=ratio_data,
        output_path=ratio_out,
        plot_cfg=plot_cfg["ratio"],
        include_true_group=include_true_group,
    )
    _plot_one(
        result_data=regret_data,
        output_path=regret_out,
        plot_cfg=plot_cfg["regret"],
        include_true_group=include_true_group,
    )

    ratio_points = _collect_points(ratio_data, include_true_group=include_true_group)
    regret_points = _collect_points(regret_data, include_true_group=include_true_group)

    ratio_non_dominated = _filter_non_dominated(ratio_points, perf_mode="max")
    regret_non_dominated = _filter_non_dominated(regret_points, perf_mode="min")

    _plot_filtered_points(
        points=ratio_non_dominated,
        output_path=ratio_filtered_out,
        plot_cfg=plot_cfg["ratio_filtered"],
    )
    _plot_filtered_points(
        points=regret_non_dominated,
        output_path=regret_filtered_out,
        plot_cfg=plot_cfg["regret_filtered"],
    )

    print(f"Saved: {ratio_out}")
    print(f"Saved: {regret_out}")
    print(f"Saved: {ratio_filtered_out}")
    print(f"Saved: {regret_filtered_out}")


if __name__ == "__main__":
    main()
