import matplotlib.pyplot as plt
import numpy as np

d_0=0.05
delta=[1.3, 0.74, 0.3]
S=[1.3,0.65,0.25,0]
colors = ['blue', 'red', 'green']
day_labels = ['day 1', 'day 2', 'day 3']
L=5
R=28
# 创建数据 - Regret 图
x = np.linspace(L, R, 1000)
y_regret = []
y_regret.append((delta[0]-S[1])*(1-1/x))  # day 1
y_regret.append((delta[1]-S[2])*(1-1/x))  # day 2
y_regret.append((delta[2]-S[3])*(1-1/x))  # day 3

# 找每个点的最大值索引
y_regret_array = np.array(y_regret)
max_indices = np.argmax(y_regret_array, axis=0)

# 创建图形 - Regret
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制三个day的线，同时标记最大值用实线
for i in range(3):
    # 找该day对应最大值的点
    is_max = (max_indices == i)
    
    # 绘制非最大值部分（虚线）
    ax.plot(x[~is_max], y_regret[i][~is_max], linestyle='--', linewidth=2, 
            color=colors[i], alpha=0.5)
    
    # 绘制最大值部分（实线）
    ax.plot(x[is_max], y_regret[i][is_max], linestyle='-', linewidth=2, 
            color=colors[i], label=day_labels[i])

# 设置图形属性
ax.set_xlabel('Price-to-cost ratio', fontsize=20)
ax.set_ylabel('Regret', fontsize=20)
# 构造自定义图例：每个day分别展示“bottleneck”和
# “non-bottleneck”两种线型，总共六条条目。
from matplotlib.lines import Line2D

legend_handles = []
legend_labels = []
for color, day in zip(colors, day_labels):
    if day == 'day 1':
        legend_handles.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
        legend_labels.append(f"{day} (bottleneck)")
    else:
        legend_handles.append(Line2D([0], [0], color=color, lw=2, linestyle='--'))
        legend_labels.append(f"{day} (non-bottleneck)")

ax.legend(legend_handles, legend_labels, fontsize=18)

ax.tick_params(labelsize=14)
ax.grid(True, alpha=0.3)

# 保存并显示
plt.tight_layout()
plt.savefig('figure_regret.png', dpi=300, bbox_inches='tight')

# 创建数据 - Ratio 图
x = np.linspace(L, R, 1000)
y_ratio = []
y_ratio.append((x*d_0+(S[1]))/(x*d_0+delta[0]))  # day 1
y_ratio.append((x*d_0+(S[2]))/(x*d_0+delta[1]))  # day 2
y_ratio.append((x*d_0+(S[3]))/(x*d_0+delta[2]))  # day 3

# 找每个点的最小值索引
y_ratio_array = np.array(y_ratio)
min_indices = np.argmin(y_ratio_array, axis=0)

# 创建图形 - Ratio
fig, ax = plt.subplots(figsize=(10, 6))

# 先把所有三条曲线画成虚线，这样不会出现跨段的连接
for i in range(3):
    ax.plot(x, y_ratio[i], linestyle='--', linewidth=2,
            color=colors[i], alpha=0.5)

# 再把每个点上的最小值段覆盖为实线
for i in range(3):
    is_min = (min_indices == i)
    if np.any(is_min):  # 只有在存在最小区间时才绘制
        ax.plot(x[is_min], y_ratio[i][is_min], linestyle='-', linewidth=2,
                color=colors[i], label=day_labels[i])

# 设置图形属性
ax.set_xlabel('Price-to-cost ratio', fontsize=20)
ax.set_ylabel('Robust competitive ratio', fontsize=20)

# 构造自定义图例：每个day分别展示“bottleneck”和
# “non-bottleneck”两种线型，总共六条条目。
from matplotlib.lines import Line2D

legend_handles = []
legend_labels = []
for color, day in zip(colors, day_labels):
    legend_handles.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
    legend_labels.append(f"{day} (bottleneck)")
    legend_handles.append(Line2D([0], [0], color=color, lw=2, linestyle='--'))
    legend_labels.append(f"{day} (non-bottleneck)")

ax.legend(legend_handles, legend_labels, fontsize=18)

ax.tick_params(labelsize=14)
ax.grid(True, alpha=0.3)

# 保存并显示
plt.tight_layout()
plt.savefig('figure_ratio.png', dpi=300, bbox_inches='tight')

