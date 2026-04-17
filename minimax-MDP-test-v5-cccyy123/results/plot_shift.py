import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
PICTURE_DIR = BASE_DIR / "picture"

# ===== Plot Config (edit these values directly) =====
FIG_SIZE = (10, 6)
LINE_WIDTH = 2.2
SHOW_MARKER = False
MARKER_STYLE = "o"
MARKER_SIZE = 3

X_LABEL = "Shift value (B)"
RATIO_Y_LABEL = "Competitive ratio"
REGRET_Y_LABEL = "Regret"

X_LABEL_SIZE = 17
Y_LABEL_SIZE = 17
X_TICK_SIZE = 17
Y_TICK_SIZE = 17
LEGEND_SIZE = 20
TITLE_SIZE = 17
SHOW_DIS_LINES = 0  # 1: show dis_1/dis_2, 0: hide them
SHOW_CONFIDENCE_RANGE = 1  # 1: show seed percentile band, 0: hide band
CONFIDENCE_ALPHA = 0.18
SEED_BAND_LOWER_PERCENT = 5.0
SEED_BAND_UPPER_PERCENT = 95.0

# Rename legend labels here.
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
	# Keep only metric curves, skip time and metadata fields.
	metric_keys = [
		k
		for k in result_data.keys()
		if k.startswith("rgt_") and not k.endswith("_time")
	]

	series = {}
	for key in metric_keys:
		if SHOW_DIS_LINES == 0 and key in {"rgt_dis_1", "rgt_dis_2", "rgt_dis"}:
			continue
		# Each entry is [group][seed]. Line uses mean; band uses sorted-seed percentiles.
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
	with open(BASE_DIR / "generate_shift.json", "r", encoding="utf-8") as f:
		generate_shift = json.load(f)
	with open(BASE_DIR / "result_ratio_shift.json", "r", encoding="utf-8") as f:
		ratio_data = json.load(f)
	with open(BASE_DIR / "result_rgt_shift.json", "r", encoding="utf-8") as f:
		regret_data = json.load(f)

	x_values = generate_shift["shift_value"]

	ratio_series = build_series(ratio_data)
	regret_series = build_series(regret_data)

	plot_curves(
		x_values=x_values,
		series=ratio_series,
		title="Shift value vs Competitive ratio",
		y_label=RATIO_Y_LABEL,
		output_name="result_ratio_shift",
	)

	plot_curves(
		x_values=x_values,
		series=regret_series,
		title="Shift value vs Regret",
		y_label=REGRET_Y_LABEL,
		output_name="result_rgt_shift",
	)


if __name__ == "__main__":
	main()