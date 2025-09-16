import os
import re
import yaml
import torch
import spacy
import numpy as np

# import numpy as np
from sklearn.metrics import ndcg_score

def auto_split_label_groups(Y, num_bins=3, min_tail_size=1):
    """
    根据标签频率自动将标签分为 head/mid/tail 三组
    Args:
        Y: ndarray (num_samples, num_labels), 0/1 标签矩阵
        num_bins: 通常为 3，分为 head/mid/tail
        min_tail_size: 控制 tail 至少包含的标签数，避免空分组
    Returns:
        label_groups: List[List[int]]
    """
    if not isinstance(Y, np.ndarray):
        Y = Y.cpu().numpy()

    label_freq = Y.sum(axis=0)  # 每个标签的频率
    label_logfreq = np.log1p(label_freq)  # 对数变换（+1 防止 log(0)）

    # 使用分位点（如 33%、66%）划分
    quantiles = np.percentile(label_logfreq, [100 * i / num_bins for i in range(1, num_bins)])
    # e.g., [q1, q2] = [33rd percentile, 66th percentile]

    label_groups = [[] for _ in range(num_bins)]
    for i, val in enumerate(label_logfreq):
        if val <= quantiles[0]:
            label_groups[0].append(i)  # Tail
        elif val <= quantiles[1]:
            label_groups[1].append(i)  # Mid
        else:
            label_groups[2].append(i)  # Head

    # 保底措施：若 tail 组太小，则合并 mid → tail
    if len(label_groups[0]) < min_tail_size and num_bins == 3:
        label_groups[1] = label_groups[0] + label_groups[1]
        label_groups[0] = []

    return label_groups

class NLP:
    def __init__(self, path):
        self.nlp = spacy.load(path, disable=['ner', 'parser', 'tagger'])
        # self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.nlp.add_pipe('sentencizer')

    def sent_tokenize(self, text):
        doc = self.nlp(text)
        sentences = [sent.string.strip() for sent in doc.sents]
        return sentences

    def word_tokenize(self, text, lower=False):  # create a tokenizer function
        if text is None:
            return text
        text = ' '.join(text.split())
        if lower:
            text = text.lower()
        toks = [tok.text for tok in self.nlp.tokenizer(text)]
        return ' '.join(toks)

class Accuracy:
    def __init__(self):
        super(Accuracy, self).__init__()
        self.total = 0
        self.acc1 = 0
        self.acc3 = 0
        self.acc5 = 0
        self.ndcg3 = 0
        self.ndcg5 = 0

    def calc(self, logits, labels):
        acc1, acc3, acc5, total, ndcg3, ndcg5 = get_accuracy(logits, labels)
        self.total += total
        self.acc1 += acc1
        self.acc3 += acc3
        self.acc5 += acc5
        self.ndcg3 += ndcg3
        self.ndcg5 += ndcg5

    def get_accuracy(self, logits, labels):
        logits = logits.detach().cpu()
        labels = labels.cpu().numpy()
        scores, indices = torch.topk(logits, k=10)

        acc1, acc3, acc5, total, ndcg3, ndcg5 = 0, 0, 0, 0, 0, 0

        for index, label in enumerate(labels):
            ndcg3 += ndcg_score(np.reshape(label, (1, -1)), np.reshape(logits, (1, -1)), k=3)
            ndcg5 += ndcg_score(np.reshape(label, (1, -1)), np.reshape(logits, (1, -1)), k=5)
            # logits_d= logits[index].numpy()
            # labels_d= label[index]
            # ndcg3 += ndcg_score(labels_d, logits_d, k=3)
            # ndcg5 += ndcg_score(labels_d, logits_d, k=5)

            label = set(np.nonzero(label)[0])

            labels = indices[index, :5].numpy()

            acc1 += len(set([labels[0]]) & label)
            acc3 += len(set(labels[:3]) & label)
            acc5 += len(set(labels[:5]) & label)

            total += 1

        return acc1, acc3, acc5, total, ndcg3, ndcg5

    def reset_acc(self):
        self.total = 0
        self.acc1 = 0
        self.acc3 = 0
        self.acc5 = 0
        self.ndcg3 = 0
        self.ndcg5 = 0

    def get_acc1(self):
        return self.acc1 / self.total

    def get_acc3(self):
        return self.acc3 / self.total / 3

    def get_acc5(self):
        return self.acc5 / self.total / 5

    def get_ndcg3(self):
        return self.ndcg3 / self.total

    def get_ndcg5(self):
        return self.ndcg5 / self.total

    def get_total(self):
        return self.total


def get_accuracy(logits, labels):
    logits = logits.detach().cpu()
    labels = labels.cpu().numpy()
    scores, indices = torch.topk(logits, k=10)

    acc1, acc3, acc5, total, ndcg3, ndcg5 = 0, 0, 0, 0, 0, 0

    for index, label in enumerate(labels):
        ndcg3 += ndcg_score(np.reshape(label, (1, -1)), np.reshape(logits[index], (1, -1)), k=3)
        ndcg5 += ndcg_score(np.reshape(label, (1, -1)), np.reshape(logits[index], (1, -1)), k=5)
        # logits_d= logits[index].numpy()
        # labels_d= label[index]
        # ndcg3 += ndcg_score(labels_d, logits_d, k=3)
        # ndcg5 += ndcg_score(labels_d, logits_d, k=5)

        label = set(np.nonzero(label)[0])

        labels = indices[index, :5].numpy()

        acc1 += len(set([labels[0]]) & label)
        acc3 += len(set(labels[:3]) & label)
        acc5 += len(set(labels[:5]) & label)

        total += 1

    return acc1, acc3, acc5, total, ndcg3, ndcg5

def save(args, savefilename, mark, model):
    # save_dir = f"./checkpoint/{datatset}"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    if args.stage == 1:
        torch.save(model.bert.state_dict(), os.path.join(savefilename, "model_file_stage1.bin"))
        with open(os.path.join(savefilename, "mark_stage1.txt"), "w", encoding="UTF-8") as mark_file:
            mark_file.write(mark)

        print(f"save stage1 model to path {savefilename} success")
    elif args.stage == 2:
        torch.save(model.state_dict(), os.path.join(savefilename, "model_file_stage2.bin"))

        with open(os.path.join(savefilename, "mark.txt"), "w", encoding="UTF-8") as mark_file:
            mark_file.write(mark)

        print(f"save stage2 model to path {savefilename} success")

def smooth_multi_label(y, num_classes, delta=0.98):
    # y: shape (batch_size, num_classes), values 0 or 1
    smooth_y = y.clone().float()
    smooth_y = smooth_y * (1 - delta + delta / num_classes) + \
               (1 - smooth_y) * (delta / num_classes)
    return smooth_y


def logits_l2_regularizer(logits: torch.Tensor, alpha: float = 1e-5, detach_center: bool = True):
    """
    logits: (B, C) 未激活分数
    alpha : 正则强度（建议 1e-5 ~ 5e-4）
    detach_center=True: 仅对去中心后的残差罚，不反传中心
    """
    m = logits.mean(dim=1, keepdim=True)                 # 每个样本自己的均值
    center = m.detach() if detach_center else m
    z_hat = logits - center
    return alpha * (z_hat ** 2).mean()


# ==== schedule & monitor helpers ====

def _get_grouped_head(model):
    """
    返回分组版头（GroupedMoE）。做鸭子判断：有 group_indices/G/gate 就认为是分组头。
    """
    head = None
    cand = []
    if hasattr(model, "moe"): cand.append(model.moe)
    if hasattr(model, "head"): cand.append(model.head)
    for h in cand:
        if h is not None and all(hasattr(h, a) for a in ["group_indices", "G", "gate"]):
            head = h; break
    return head  # 可能是 None（比如用的不是分组头时）

def _build_schedule(args, head, steps_per_epoch):
    """
    构造训练日程（step 粒度）。你也可以把这些值放到 args 里传入。
    这里给默认值：前 15% steps Dense；15%~40% 半稀疏；其后目标稀疏。
    """
    total_steps = args.epochs * steps_per_epoch
    dense_steps = getattr(args, "dense_steps", max(1000, int(0.15 * total_steps)))
    semi_steps  = getattr(args, "semi_steps",  max(3000, int(0.40 * total_steps)))
    # dense_steps = 2000
    # semi_steps = 5000
    target_topk = getattr(args, "target_topk", 5)  # B数据集建议=5（G≈62）
    sched = {
        "dense":  dense_steps,
        "semi":   semi_steps,
        "temp_dense": 2.0,
        "temp_semi":  1.5,
        "temp_target": 1.0,
        "balance_semi":  0.005,
        "balance_target": 0.01,
        "target_topk": min(max(1, target_topk), head.G if head is not None else target_topk),
    }
    return sched

def _apply_schedule(head, step, sched):
    if head is None: return
    if step < sched["dense"]:
        head.topk = head.G                 # Dense 预热
        head.gate_temp = sched["temp_dense"]
        head.balance_coef = 0.0
    elif step < sched["semi"]:
        head.topk = max(8, int(0.25 * head.G))  # 半稀疏，约 1/4 组
        head.gate_temp = sched["temp_semi"]
        head.balance_coef = sched["balance_semi"]
    else:
        head.topk = sched["target_topk"]   # 目标
        head.gate_temp = sched["temp_target"]
        head.balance_coef = sched["balance_target"]

def _log_gate_stats(head):
    """
    返回 (max_importance, entropy)；若不可用返回 None。
    """
    if head is None: return None
    gp = getattr(head, "_last_gate_probs", None)  # [B,G]
    if gp is None: return None
    with torch.no_grad():
        importance = gp.mean(0)  # [G]
        entropy = (-(gp.clamp_min(1e-8).log()) * gp).sum(-1).mean()
        return float(importance.max().item()), float(entropy.item())
