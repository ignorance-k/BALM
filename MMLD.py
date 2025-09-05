import torch
from transformers import BertConfig, BertModel
import torch.nn as nn
import torch.nn.functional as F


class Base(nn.Module):
    """Base分支"""
    def __init__(self, label_number, bert_hidden_size, feature_layers=5, dropout=0.5):
        super().__init__()
        self.bert_hidden_size = bert_hidden_size
        self.feature_layers = feature_layers
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(feature_layers * self.bert_hidden_size, label_number)
        nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, hidden_states):
        out = torch.cat([hidden_states[-i][:, 0] for i in range(1, self.feature_layers + 1)], dim=-1)
        logits = self.linear(self.dropout(out))
        return logits



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union

def _normalize_groups(label_groups: Union[List[List[int]], Dict[str, List[int]]]):
    if isinstance(label_groups, dict):
        # 如果你想按插入顺序，用 .values()；这里用 keys 排序保证可复现
        try:
            # 保留插入顺序（Py3.7+）
            groups = list(label_groups.values())
        except Exception:
            groups = [label_groups[k] for k in sorted(label_groups.keys())]
    else:
        groups = label_groups
    # 转成长整型张量列表
    norm = []
    for g in groups:
        idx = torch.tensor(g, dtype=torch.long)
        assert idx.dim() == 1, "每个组应为一维索引列表"
        norm.append(idx)
    return norm

class GroupedMoE(nn.Module):
    """
    9_3
    分组版 MoE：每个 expert 只预测其对应标签簇的logits，然后 scatter_add 回全维度
    - 输入: h [B, D]
    - 输出: logits [B, C], balance_loss (标量)
    """
    def __init__(
        self,
        num_labels: int,  # 全量标签数 C
        in_dim: int,
        label_groups: Union[List[List[int]], Dict[str, List[int]]],
        topk: int = 2,                  # Top-K 选多少个“组专家”
        dropout: float = 0.1,
        expert_hidden: int = 0,         # 0=线性专家；>0=两层MLP
        use_residual_base: bool = True, # 是否叠加全维Base头
        base_weight: float = 1.0,
        moe_weight: float = 1.0,
        # 门控与正则
        gate_temp: float = 1.0,
        noisy_gate: bool = False,
        gate_noise_std: float = 0.0,
        balance_coef: float = 0.01      # 负载均衡强度（importance/load 双约束）
    ):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.gate_temp = gate_temp
        self.noisy_gate = noisy_gate
        self.gate_noise_std = gate_noise_std
        self.balance_coef = balance_coef
        self.num_labels = num_labels
        self.moe_weight = moe_weight
        self.base_weight = base_weight

        # 归一化组索引
        groups = _normalize_groups(label_groups)
        self.group_indices: List[torch.Tensor] = groups
        self.G = len(self.group_indices)
        assert self.G >= 1, "至少需要一个分组"
        self.topk = min(max(1, topk), self.G)  # 安全裁剪

        # 专家：每个组一个头，输出维度=该组大小
        def make_expert(out_dim: int):
            if expert_hidden and expert_hidden > 0:
                block = nn.Sequential(
                    nn.Linear(in_dim, expert_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden, out_dim),
                )
                for m in block:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                return block
            else:
                layer = nn.Linear(in_dim, out_dim)
                nn.init.xavier_uniform_(layer.weight)
                return layer

        self.experts = nn.ModuleList([make_expert(len(idx)) for idx in self.group_indices])

        # 组级门控：预测每一组的重要性
        self.gate = nn.Linear(in_dim, self.G)
        nn.init.xavier_uniform_(self.gate.weight)

        # 残差Base：全维线性头（可关）
        self.base = nn.Linear(in_dim, num_labels) if use_residual_base else None
        if self.base is not None:
            nn.init.xavier_uniform_(self.base.weight)

        # 记录用：调试门控分布
        self._last_gate_probs = None

    def forward(self, h: torch.Tensor, return_balance_loss: bool = True):
        """
        h: [B, D]
        """
        B = h.size(0)
        device = h.device
        h = self.drop(h)

        # 组级门控
        gate_logits = self.gate(h)                           # [B, G]
        if self.noisy_gate and self.training and self.gate_noise_std > 0:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.gate_noise_std
        gate_probs = torch.softmax(gate_logits / max(self.gate_temp, 1e-6), dim=-1)   # [B, G]
        self._last_gate_probs = gate_probs.detach()

        # Top-K 稀疏化（对被选中的组重新归一化）
        if self.topk < self.G:
            topv, topi = torch.topk(gate_probs, self.topk, dim=-1)        # [B, K]
            # 构造每样本的组权重矩阵 W: [B, G]
            W = torch.zeros_like(gate_probs)                               # [B, G]
            W.scatter_(1, topi, topv)
            # 归一化只在选中组内做（保持期望一致）
            Z = W.sum(dim=1, keepdim=True).clamp_min(1e-8)                 # [B, 1]
            W = W / Z
        else:
            W = gate_probs                                                 # Dense

        # 聚合：对每个组的专家输出做带权 scatter_add 到全维 logits
        logits = torch.zeros(B, self.num_labels, device=device)
        for g, idx in enumerate(self.group_indices):
            idx = idx.to(device)                                           # [m]
            out_g = self.experts[g](h)                                     # [B, m]
            w = W[:, g].unsqueeze(1)                                       # [B, 1]
            contrib = out_g * w                                            # [B, m]
            # 扩展索引到 [B, m] 以做 batch scatter_add
            scatter_idx = idx.view(1, -1).expand(B, -1)                    # [B, m]
            logits = logits.scatter_add(1, scatter_idx, contrib)           # [B, C]

        # 残差 Base（可选）
        if self.base is not None and self.base_weight != 0.0:
            logits = self.moe_weight * logits + self.base_weight * self.base(h)
        else:
            logits = self.moe_weight * logits

        # 负载均衡（importance/load 双约束）
        if self.training and self.balance_coef > 0:
            importance = gate_probs.mean(dim=0)                             # [G]（门控的平均概率）
            load = W.mean(dim=0)                                            # [G]（实际被选中的平均权重）
            uniform = torch.full_like(importance, 1.0 / self.G)
            loss_imp = torch.mean((importance - uniform) ** 2)
            loss_load = torch.mean((load - uniform) ** 2)
            balance_loss = self.balance_coef * (loss_imp + loss_load)
        else:
            balance_loss = torch.tensor(0.0, device=device)

        return (logits, balance_loss) if return_balance_loss else (logits, torch.tensor(0.0, device=device))



# class MoE(nn.Module):
#     """
#     9_2：两次市lr的不同
#     改进版稀疏 MoE 头：
#     - Dense 预热（前若干 step 全专家加权）
#     - 门控温度退火 + 可选噪声
#     - 负载均衡：importance/load 双约束
#     - router z-loss 与熵正则（抑制门控过于尖锐）
#     - 可选残差 base 及其权重
#     """
#     def __init__(
#         self,
#         label_number,
#         in_dim,
#         num_experts=4,
#         topk=2,
#         balance_coef=0.02,
#         expert_hidden=512,
#         dropout=0.5,
#         use_residual_base=True,
#         base_weight=1.0,          # 残差基线权重
#         moe_weight=1.0,           # MoE 主路权重
#         # 训练稳定性
#         dense_warmup_steps=2000,  # 预热步数：前 N step 用 Dense
#         gate_temp_init=2.0,       # 初始温度（>1 更“软”）
#         gate_temp_min=1.0,        # 最小温度
#         temp_anneal_steps=4000,   # 温度线性退火到 min 的步数
#         noisy_gate=True,
#         gate_noise_std=0.5,
#         # 正则
#         router_zloss_coef=1e-3,   # z-loss 抑制 logsumexp 过大
#         gate_entropy_coef=1e-3,   # 熵正则：鼓励更均匀的门控
#     ):
#         super().__init__()
#         assert 1 <= topk <= num_experts
#         self.num_experts = num_experts
#         self.topk = topk
#         self.balance_coef = balance_coef
#         self.label_number = label_number
#
#         self.noisy_gate = noisy_gate
#         self.gate_noise_std = gate_noise_std
#
#         self.dense_warmup_steps = dense_warmup_steps
#         self.gate_temp_init = gate_temp_init
#         self.gate_temp_min = gate_temp_min
#         self.temp_anneal_steps = temp_anneal_steps
#         self.router_zloss_coef = router_zloss_coef
#         self.gate_entropy_coef = gate_entropy_coef
#
#         self.moe_weight = moe_weight
#         self.base_weight = base_weight
#
#         # 计步器（非持久 buffer，DDP 也安全）
#         self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)
#
#         def make_expert():
#             if expert_hidden and expert_hidden > 0:
#                 block = nn.Sequential(
#                     nn.Linear(in_dim, expert_hidden),
#                     nn.GELU(),
#                     nn.Dropout(dropout),
#                     nn.Linear(expert_hidden, label_number),
#                 )
#                 for m in block:
#                     if isinstance(m, nn.Linear):
#                         nn.init.xavier_uniform_(m.weight)
#                 return block
#             else:
#                 layer = nn.Linear(in_dim, label_number)
#                 nn.init.xavier_uniform_(layer.weight)
#                 return layer
#
#         self.experts = nn.ModuleList([make_expert() for _ in range(num_experts)])
#         self.gate = nn.Linear(in_dim, num_experts)
#         nn.init.xavier_uniform_(self.gate.weight)
#
#         self.base = nn.Linear(in_dim, label_number) if use_residual_base else None
#         if self.base is not None:
#             nn.init.xavier_uniform_(self.base.weight)
#
#         self.drop = nn.Dropout(dropout)
#
#     def _current_temp(self):
#         # 线性退火 gate 温度
#         if self.temp_anneal_steps <= 0:
#             return self.gate_temp_min
#         step = min(int(self._step.item()), self.temp_anneal_steps)
#         t = self.gate_temp_init + (self.gate_temp_min - self.gate_temp_init) * (step / self.temp_anneal_steps)
#         return max(self.gate_temp_min, float(t))
#
#     def forward(self, h, return_balance_loss=False):
#         # 计步
#         if self.training:
#             self._step += 1
#
#         h = self.drop(h)                                 # [B, D]
#         gate_logits = self.gate(h)                       # [B, E]
#
#         # z-loss（SwitchTransformer 风格）：抑制 logsumexp 过大，防 saturate
#         z = torch.logsumexp(gate_logits, dim=-1)         # [B]
#         z_loss = (z ** 2).mean() * self.router_zloss_coef if self.training else torch.tensor(0., device=h.device)
#
#         # 温度退火 + 噪声门控
#         temp = self._current_temp()
#         if self.noisy_gate and self.training:
#             gate_logits = gate_logits + torch.randn_like(gate_logits) * self.gate_noise_std
#         gate_logits = gate_logits / temp
#         gate_probs = torch.softmax(gate_logits, dim=-1)  # [B, E]
#
#         # Dense 预热
#         use_dense = self.training and (int(self._step.item()) <= self.dense_warmup_steps)
#         k = self.num_experts if use_dense else self.topk
#
#         if k < self.num_experts:
#             topv, topi = torch.topk(gate_probs, k, dim=-1)                # [B, K]
#             topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-8)         # 归一化
#
#             outs = torch.stack([expert(h) for expert in self.experts], dim=2)  # [B, C, E]
#             gather = outs.gather(2, topi.unsqueeze(1).expand(-1, outs.size(1), -1))  # [B, C, K]
#             moe_logits = (gather * topv.unsqueeze(1)).sum(dim=2)           # [B, C]
#         else:
#             outs = torch.stack([expert(h) for expert in self.experts], dim=2)      # [B, C, E]
#             moe_logits = (outs * gate_probs.unsqueeze(1)).sum(dim=2)               # [B, C]
#
#         logits = self.moe_weight * moe_logits
#         if self.base is not None and self.base_weight != 0.0:
#             logits = logits + self.base_weight * self.base(h)
#
#         # 负载均衡：importance/load 双约束 + 熵正则
#         if self.training and self.balance_coef > 0:
#             importance = gate_probs.mean(dim=0)                                # [E]
#             if k < self.num_experts:
#                 B = gate_probs.size(0)
#                 counts = torch.bincount(topi.reshape(-1), minlength=self.num_experts).float()
#                 load = counts / (B * k + 1e-8)                                 # [E]
#             else:
#                 load = importance.detach()
#
#             uniform = torch.full_like(importance, 1.0 / self.num_experts)
#             loss_imp = torch.mean((importance - uniform) ** 2)
#             loss_load = torch.mean((load - uniform) ** 2)
#             balance_loss = self.balance_coef * (loss_imp + loss_load)
#         else:
#             balance_loss = torch.tensor(0.0, device=h.device)
#
#         # 熵正则（鼓励更均匀门控）
#         if self.training and self.gate_entropy_coef > 0:
#             entropy = (-gate_probs.clamp_min(1e-8).log() * gate_probs).sum(dim=-1).mean()
#             balance_loss = balance_loss + (- self.gate_entropy_coef * entropy)
#
#         # z-loss
#         balance_loss = balance_loss + z_loss
#
#         return (logits, balance_loss) if return_balance_loss else (logits, torch.tensor(0.0, device=h.device))




# class MoE(nn.Module):
#     """
#      9_1 日
#     稀疏 MoE 多专家分类头（Top-K gating + 负载均衡）
#     - 输入: h [B, D]
#     - 输出: logits [B, C], balance_loss (标量)
#     """
#     def __init__(
#         self,
#         label_number,
#         in_dim,
#         num_experts=6,
#         topk=2,
#         balance_coef=0.02,
#         expert_hidden=512,
#         dropout=0.5,
#         use_residual_base=True,
#         noisy_gate=False,             # 可选：加噪门控更平滑
#         gate_noise_std=0.5,
#     ):
#         super().__init__()
#         assert 1 <= topk <= num_experts, "topk 必须在 [1, num_experts] 范围内"
#         self.num_experts = num_experts
#         self.topk = topk
#         self.balance_coef = balance_coef
#         self.label_number = label_number
#         self.noisy_gate = noisy_gate
#         self.gate_noise_std = gate_noise_std
#
#         # Expert 定义：两层 MLP 或线性
#         def make_expert():
#             if expert_hidden and expert_hidden > 0:
#                 block = nn.Sequential(
#                     nn.Linear(in_dim, expert_hidden),
#                     nn.GELU(),
#                     nn.Dropout(dropout),
#                     nn.Linear(expert_hidden, label_number),
#                 )
#                 # 初始化略微保守
#                 for m in block:
#                     if isinstance(m, nn.Linear):
#                         nn.init.xavier_uniform_(m.weight)
#                 return block
#             else:
#                 layer = nn.Linear(in_dim, label_number)
#                 nn.init.xavier_uniform_(layer.weight)
#                 return layer
#
#         self.experts = nn.ModuleList([make_expert() for _ in range(num_experts)])
#         self.gate = nn.Linear(in_dim, num_experts)
#         nn.init.xavier_uniform_(self.gate.weight)
#
#         self.base = nn.Linear(in_dim, label_number) if use_residual_base else None
#         if self.base is not None:
#             nn.init.xavier_uniform_(self.base.weight)
#
#         self.drop = nn.Dropout(dropout)
#
#     def forward(self, h, return_balance_loss=False):
#         """
#         h: [B, D]
#         """
#         h = self.drop(h)                                   # [B, D]
#         gate_logits = self.gate(h)                         # [B, E]
#
#         if self.noisy_gate and self.training:
#             gate_logits = gate_logits + torch.randn_like(gate_logits) * self.gate_noise_std
#
#         gate_probs = torch.softmax(gate_logits, dim=-1)    # [B, E]
#
#         # --- 稀疏 Top-K 聚合 ---
#         if self.topk < self.num_experts:
#             topv, topi = torch.topk(gate_probs, self.topk, dim=-1)      # [B, K]
#             # 归一化所选专家权重，避免数值偏移
#             topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-8)       # [B, K]
#
#             # 先计算所有专家输出，再 gather（实现简单且 GPU 上足够快）
#             # outs: [E][B, C] -> [B, C, E]
#             outs = torch.stack([expert(h) for expert in self.experts], dim=2)  # [B, C, E]
#
#             # 按选中的 expert 索引并加权
#             gather = outs.gather(2, topi.unsqueeze(1).expand(-1, outs.size(1), -1))  # [B, C, K]
#             logits = (gather * topv.unsqueeze(1)).sum(dim=2)                         # [B, C]
#         else:
#             # 全部专家加权（Dense）
#             outs = torch.stack([expert(h) for expert in self.experts], dim=2)        # [B, C, E]
#             logits = (outs * gate_probs.unsqueeze(1)).sum(dim=2)                     # [B, C]
#
#         if self.base is not None:
#             logits = logits + self.base(h)
#
#         # --- 负载均衡（期望每个 expert 被均匀使用）---
#         if self.training and self.balance_coef > 0:
#             # importance: 每个 expert 的平均门控概率
#             importance = gate_probs.mean(dim=0)                                      # [E]
#             # load 估计：近似为被选中的频率（Top-K 时只统计被选中的专家）
#             if self.topk < self.num_experts:
#                 # one-hot 统计：每个样本的 topi 标记为 1/K，再对 batch 求均值
#                 B, K = gate_probs.size(0), self.topk
#                 load = torch.zeros(self.num_experts, device=h.device)                # [E]
#                 # 统计每个 expert 在 topk 中出现的次数（期望）
#                 counts = torch.bincount(topi.reshape(-1), minlength=self.num_experts).float()  # [E]
#                 load = counts / (B * K + 1e-8)
#             else:
#                 load = importance.detach()
#
#             # Switch/Balance 风格损失：使 importance 和 load 都接近均匀分布
#             uniform = torch.full_like(importance, 1.0 / self.num_experts)
#             loss_imp = torch.mean((importance - uniform) ** 2)
#             loss_load = torch.mean((load - uniform) ** 2)
#             balance_loss = self.balance_coef * (loss_imp + loss_load)
#         else:
#             balance_loss = torch.tensor(0.0, device=h.device)
#
#         return (logits, balance_loss) if return_balance_loss else (logits, torch.tensor(0.0, device=h.device))




# class MoE(nn.Module):
#     """MoE分支
#       分组 - 8_29
#     """
#     def __init__(self, label_groups, label_number, feature_layers=5, bert_hidden_size=768, dropout=0.5):
#         super().__init__()
#         self.label_groups = label_groups
#         self.label_number = label_number
#         self.feature_layers = feature_layers
#         self.dropout = nn.Dropout(dropout)
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(feature_layers * bert_hidden_size, bert_hidden_size),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(bert_hidden_size, len(group))
#             ) for group in label_groups if len(group) > 0
#         ])
#         self.non_empty_groups = [group for group in label_groups if len(group) > 0]
#
#     def forward(self, hidden_states):
#         out = torch.cat([hidden_states[-i][:, 0] for i in range(1, self.feature_layers + 1)], dim=-1)
#         out = self.dropout(out)
#
#         expert_outputs = [expert(out) for expert in self.experts]
#         batch_size = out.size(0)
#         num_labels = sum(len(g) for g in self.non_empty_groups)
#         logits = torch.zeros(batch_size, num_labels, device=out.device)
#         for group_idx, group in enumerate(self.non_empty_groups):
#             logits[:, group] = expert_outputs[group_idx]
#         return logits


class MMLD(nn.Module):
    def __init__(self, label_number, feature_layers, bert_path, label_groups=None, use_moe=True, dropout=0.5):
        super(MMLD, self).__init__()
        # 加载bert
        config = BertConfig.from_pretrained(bert_path)
        config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        for param in self.bert.parameters():
            param.data = param.data.contiguous()
        self.bert_hidden_size = self.bert.config.hidden_size

        # model参数
        self.label_number = label_number
        self.feature_layers = feature_layers
        self.label_groups = label_groups
        self.use_moe = use_moe
        self.dropout = dropout

        if self.use_moe:
            assert label_groups is not None, 'label_groups must be provided for MoE'
            # self.moe = MoE(self.label_groups, self.label_number, self.feature_layers, self.bert_hidden_size, self.dropout)
            self.moe = GroupedMoE(self.label_number, self.feature_layers * self.bert_hidden_size, self.label_groups)
        else:
            self.base = Base(self.label_number, self.bert_hidden_size, self.feature_layers, self.dropout)

    def _make_feature(self, hidden_states):
        """拼接 BERT 后 feature_layers 层的 [CLS] 作为 MoE/Base 的输入向量"""
        cls_stack = [hidden_states[-i][:, 0] for i in range(1, self.feature_layers + 1)]
        h = torch.cat(cls_stack, dim=-1)  # [B, D]
        return h

    def forward(self, batchs):
        # bert初步处理
        bert_out = self.bert(input_ids=batchs['input_ids'], attention_mask=batchs['attention_mask'], token_type_ids=batchs['token_type_ids'], output_hidden_states=True)
        hidden_states = bert_out.hidden_states
        if self.use_moe:
            h = self._make_feature(hidden_states)                # [B, D]
            logits, balance_loss = self.moe(h, True)
        else:
            logits = self.base(hidden_states)
            balance_loss = torch.tensor(0.0, device=logits.device)

        return logits, balance_loss
