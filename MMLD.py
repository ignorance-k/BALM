from transformers import BertConfig, BertModel
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import List, Dict, Union

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
    AAPD参数：tok:2  dropout:0.1  expert_hidden:0  use_residual_base:True  base_weight: 1.0
    moe_weight:1.0  gate_temp:1.0  noisy_gate:False   gate_noise_std:0.0  balance_coef:0.01
    """
    def __init__(
        self,
        num_labels: int,  # 全量标签数 C
        in_dim: int,
        label_groups: Union[List[List[int]], Dict[str, List[int]]],
        topk: int = 2,                  # Top-K 选多少个“组专家”, AAPD:2, rcv1:3
        dropout: float = 0.1,
        expert_hidden: int = 0,         # 0=线性专家；>0=两层MLP
        use_residual_base: bool = True, # 是否叠加全维Base头
        base_weight: float = 1.0,
        moe_weight: float = 1.0,
        # 门控与正则
        gate_temp: float = 1.0,            #1.0, 2.0
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

        # === 集合 AAPD ===
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

        # # === 聚合 -- 提速（rcv1） ===
        # # 选出本 batch 实际用到的组（W[:, g] > 0）
        # if self.topk < self.G:
        #     selected = (W.max(dim=0).values > 0).nonzero(as_tuple=False).squeeze(1)  # [U]
        # else:
        #     selected = torch.arange(self.G, device=device)
        # logits = torch.zeros(B, self.num_labels, device=device)
        # for g in selected.tolist():
        #     idx = self.group_indices[g].to(device)  # [m]
        #     out_g = self.experts[g](h)  # [B, m]
        #     w = W[:, g].unsqueeze(1)  # [B, 1]
        #     contrib = out_g * w  # [B, m]
        #     scatter_idx = idx.view(1, -1).expand(B, -1)  # [B, m]
        #     logits = logits.scatter_add(1, scatter_idx, contrib)  # [B, C]


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
