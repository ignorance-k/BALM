import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体
matplotlib.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# 示例数据
labels = ["topk-1", "topk-2", "topk 3"]
data1 = [82.48, 82.87, 82.66]   # 系列A
data2 = [85.82, 86.45, 85.96]   # 系列B

x = np.arange(len(labels))
width = 0.35

# 创建画布
fig, ax = plt.subplots(figsize=(6, 4))

# 浅色系
bars1 = ax.bar(x - width/2, data1, width, label="N@3", color="#aec7e8", edgecolor="black")
bars2 = ax.bar(x + width/2, data2, width, label="N@5", color="#ffbb78", edgecolor="black")

# 设置坐标轴
ax.set_ylabel("Score")
ax.set_xlabel("Top-k")
ax.set_xticks(x)
ax.set_xticklabels(labels)

# **缩小 y 轴范围，放大差距**
ax.set_ylim(80, 88)

# 添加网格
ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

# 添加图例
ax.legend(frameon=False)

# 在柱子上方标注数值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),  # 偏移
                    textcoords="offset points",
                    ha="center", va="bottom")

# 紧凑布局 & 保存
plt.tight_layout()
plt.savefig("grouped_bar_chart_zoom.pdf", format="pdf")
plt.show()
