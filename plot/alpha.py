import matplotlib.pyplot as plt
import matplotlib

# 设置全局字体
matplotlib.rcParams.update({
    "font.size": 10,          # 全局字体大小
    "axes.labelsize": 10,     # 坐标轴标题大小
    "xtick.labelsize": 10,    # x轴刻度字体
    "ytick.labelsize": 10,    # y轴刻度字体
    "legend.fontsize": 10,    # 图例字体
    "pdf.fonttype": 42,       # 确保文字以文本形式保存，而不是路径
    "ps.fonttype": 42
})

# 示例数据
x = [1e-5, 2e-5, 3e-5, 4e-5]
y1 = [86.60, 86.60, 86.30, 86.60]
y2 = [82.87, 82.65, 82.39, 81.60]
y3 = [86.45, 85.89, 85.76, 84.96]


styles = [
    {"color": "#1f77b4", "linestyle": "-",  "marker": "o"},   # 深蓝
    {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},   # 橙色
    {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},   # 绿色
]

# 创建画布
plt.figure(figsize=(6, 4))

# 绘制五组折线
plt.plot(x, y1, label="P@1", **styles[0], linewidth=1.2, markersize=4)
plt.plot(x, y2, label="N@3", **styles[1], linewidth=1.2, markersize=4)
plt.plot(x, y3, label="N@5", **styles[2], linewidth=1.2, markersize=4)


# 设置标题和坐标轴
plt.xlabel("α-2")
plt.ylabel("Score")

# 添加网格（细虚线，低对比度）
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# 添加图例
plt.legend(frameon=False, loc="best")  # 去掉图例边框

# 紧凑布局 & 保存为PDF
plt.tight_layout()
plt.savefig("five_groups_line_chart.pdf", format="pdf")

# 显示图像
plt.show()
