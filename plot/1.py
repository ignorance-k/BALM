import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以复现
np.random.seed(42)

# 模拟 logits 分布
# Before regularization: larger mean diff, larger variance
pos_before = np.random.normal(loc=2.0, scale=1.2, size=2000)
neg_before = np.random.normal(loc=0.0, scale=1.2, size=2000)

# After regularization: smaller mean diff, smaller variance
pos_after = np.random.normal(loc=1.0, scale=0.6, size=2000)
neg_after = np.random.normal(loc=0.3, scale=0.6, size=2000)

# 计算指标
def compute_metrics(pos, neg):
    L = np.mean(pos) - np.mean(neg)
    sigma = np.std(np.concatenate([pos, neg]))
    r = sigma / (L + 1e-8)
    return L, sigma, r

L_before, sigma_before, r_before = compute_metrics(pos_before, neg_before)
L_after, sigma_after, r_after = compute_metrics(pos_after, neg_after)

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# --- 左: before ---
axes[0].hist(pos_before, bins=50, density=True, alpha=0.5, color="blue", label="Positive logits")
axes[0].hist(neg_before, bins=50, density=True, alpha=0.5, color="red", label="Negative logits")
axes[0].axvline(np.mean(pos_before), color="blue", linestyle="--")
axes[0].axvline(np.mean(neg_before), color="red", linestyle="--")
axes[0].set_title("Before Regularization")
axes[0].set_xlabel("Logits")
axes[0].legend()

# --- 右: after ---
axes[1].hist(pos_after, bins=50, density=True, alpha=0.5, color="blue", label="Positive logits")
axes[1].hist(neg_after, bins=50, density=True, alpha=0.5, color="red", label="Negative logits")
axes[1].axvline(np.mean(pos_after), color="blue", linestyle="--")
axes[1].axvline(np.mean(neg_after), color="red", linestyle="--")
axes[1].set_title("After Regularization")
axes[1].set_xlabel("Logits")
axes[1].legend()

# 去除刻度
for ax in axes:
    ax.tick_params(axis='both', which='both', length=0)  # 去除刻度线
    ax.set_xticks([])  # 去除x轴刻度标签
    ax.set_yticks([])  # 去除y轴刻度标签

plt.suptitle("Effect of Centered L2 Regularization on Logits Distribution", fontsize=14)
plt.tight_layout()
plt.show()
