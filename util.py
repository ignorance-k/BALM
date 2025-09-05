import os
import re
import yaml
import torch
import spacy
import numpy as np

# import numpy as np
from sklearn.metrics import ndcg_score

label_dict_change = {
    "cmp-lg": "cmp-lg",
    "cond-mat.dis-nn": "Condensed Matter Disordered Systems Neural Networks",
    "cond-mat.stat-mech": "Condensed Matter Statistical Mechanics",
    "cs.AI": "Computer Artificial Intelligence",
    "cs.CC": "Computer Computational Complexity",
    "cs.CE": "Computer Computational Engineering Finance Science",
    "cs.CG": "Computer Computational Geometry",
    "cs.CL": "Computer Computation Language",
    "cs.CR": "Computer Cryptography Security",
    "cs.CV": "Computer Vision Pattern Recognition",
    "cs.CY": "Computers Society",
    "cs.DB": "Computer Databases",
    "cs.DC": "Computer Distributed Parallel Cluster Computing",
    "cs.DL": "Computer Digital Libraries",
    "cs.DM": "Computer Discrete Mathematics",
    "cs.DS": "Computer Data Structures Algorithms",
    "cs.FL": "Computer Formal Languages Automata Theory",
    "cs.GT": "Computer Science Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Computer Information Retrieval",
    "cs.IT": "Computer Information Theory",
    "cs.LG": "Computer Machine Learning",
    "cs.LO": "Logic in Computer Science",
    "cs.MA": "Computer Multiagent Systems",
    "cs.MM": "Computer Multimedia",
    "cs.MS": "Computer Mathematical Software",
    "cs.NA": "Computer Numerical Analysis",
    "cs.NE": "Computer eural Evolutionary Computing",
    "cs.NI": "Computer Networking Internet Architecture",
    "cs.PF": "Computer Performance",
    "cs.PL": "Computer Programming Languages",
    "cs.RO": "Computer Robotics",
    "cs.SC": "Computer Symbolic Computation",
    "cs.SE": "Computer Software Engineering",
    "cs.SI": "Computer Social Information Networks",
    "cs.SY": "Computer Systems Control",
    "math.CO": "Combinatorics",
    "math.IT": "Information Theory",
    "math.LO": "Logic",
    "math.NA": "Numerical Analysis",
    "math.NT": "Number Theory",
    "math.OC": "Optimization Control",
    "math.PR": "Probability",
    "math.ST": "Statistics Theory",
    "nlin.AO": "NMathematics onlinear Adaptation Self-Organizing Systems",
    "physics.data-an": "Physics Data Analysis Statistics Probability",
    "physics.soc-ph": "Physics Society",
    "q-bio.NC": "Quantitative Biology Neurons Cognition",
    "q-bio.QM": "Biology Quantitative Methods",
    "quant-ph": "Quantum Physics",
    "stat.AP": "Statistics Applications",
    "stat.ME": "Statistics Methodology",
    "stat.ML": "Statistics Machine Learning",
    "stat.TH": "Statistics Theory"
}

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