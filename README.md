# BALM

Balancing Logits Magnitude and Experts: A Unified Approach to Long-Tailed Multi-Label Text Classification

LTMLTC In practice, labels usually follow a long-tail distribution, where most samples are concentrated in a few dominant classes, while the rest are scattered among many rare classes (tail labels).

![image-20251216161251028](/image-20251216161251028.png)

![image-20251216161300022](/image-20251216161300022.png)


## Complexity Analysis

### 1) Greedy Grouping (Algorithm 1)

This grouping is a **one-time offline** preprocessing step.

- Sorting labels by frequency: $O(C \log C)$.
- Assigning each label to a group:
  - naive scan to find the minimum-load group: $O(CG)$,
  - or with a heap to maintain group loads: $O(C \log G)$.

Here, $C$ is the number of labels and $G$ is the number of groups (often $G \approx \lceil C/T \rceil$ if each group has capacity $T$).

### 2) GSER Forward Cost (Sparse MoE at Group Level)

Let the encoder hidden size be $D$, activated group count be $K$, and group size be $m_g$ (with $\sum_{g} m_g = C$).

- **Gating/router scoring** (e.g., linear router): $O(GD)$.

- **Top-$K$ selection**: typically $O(G)$ (partial selection) or $O(G \log K)$ depending on implementation.

- Experts forward (only activated groups)

  $$\sum_{g \in \text{Top-}K} O(D m_g).$$

If groups are approximately balanced ($m_g \approx T$), this becomes $O(KDT)$.

- **Scatter back to full label space**: $O(\sum_{g \in \text{Top-}K} m_g) \approx O(KT)$.
- **Optional residual base head** $B(h) = W_b h + b_b$: adds a dense $O(CD)$ term.

**Interpretation.** Without the residual dense head, the main expert-side cost scales with $K$ and group capacity $T$ (sparse). With the dense head enabled, the inference cost additionally includes $O(CD)$.

### 3) ILMR Overhead

Computing $R_{\text{ILMR}}$ requires processing logits over the batch and label dimensions, typically $O(BC)$, which is the same order as the per-label BCE computation and usually adds a small constant-factor overhead.

------

### MoE Complexity (General Comparison)

- Dense MoE (activate all experts):

  $$\sum_{g=1}^{G} O(D m_g) = O(DC),$$

  plus router cost $O(GD)$. This is comparable to a full dense classifier over all labels.

- Sparse MoE (GSER, activate Top-$K$ groups):

  $$\sum_{g \in \text{Top-}K} O(D m_g) \approx O(KDT),$$

  plus router cost $O(GD)$. When $K \ll G$, this yields substantial sparsity while load-balancing helps prevent expert collapse.
