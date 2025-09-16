import json
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from util import auto_split_label_groups, label_dict_change
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import Counter
import numpy as np
from functools import partial
import math
import os

class AppDataset(Dataset):
    def __init__(self, dataset, tokenizer, data_type, is_MLGN):
        super(AppDataset, self).__init__()

        with open(f"./data/{dataset}/processed_{data_type}.json", "r", encoding="utf-8") as data_file, \
                open(f"./data/{dataset}/label_dict.pkl", "rb") as dict_file:
            self.json_data_list = json.load(data_file)
            self.label_dict = pickle.load(dict_file)
            self.tokenizer = tokenizer
            self.label_groups = self.label_group()
            self.is_MLGN = is_MLGN

    def __len__(self):
        return len(self.json_data_list)

    def __getitem__(self, item):
        json_data = self.json_data_list[item]

        text_encoder = self.tokenizer.encode(json_data['text'], add_special_tokens=True, max_length=512,
                                             truncation=True)
        label_one_hot = torch.zeros(len(self.label_dict)).scatter(0,
                                                                  torch.tensor([self.label_dict[data] for data in
                                                                                json_data["label"]]),
                                                                  torch.tensor([1.0 for _ in json_data["label"]]))

        return text_encoder, label_one_hot

    def label_group(self):
        label_hot = []
        for json_data in self.json_data_list:
            label_one_hot = torch.zeros(len(self.label_dict)).scatter(0,
                                                                      torch.tensor([self.label_dict[data] for data in
                                                                                    json_data["label"]]),
                                                                      torch.tensor([1.0 for _ in json_data["label"]]))
            label_hot.append(label_one_hot)
        # 将tensor列表转换为numpy数组
        label_hot = torch.stack(label_hot).numpy()
        return auto_split_label_groups(label_hot)


def padding(inputs, pad_idx, max_len=None):
    if max_len is None:
        lengths = [len(inp) for inp in inputs]
        max_len = max(lengths)
    padded_inputs = torch.as_tensor([inp + [pad_idx] * (max_len - len(inp)) for inp in inputs], dtype=torch.long)
    # mask
    masks = torch.as_tensor([[1] * len(inp) + [0] * (max_len - len(inp)) for inp in inputs], dtype=torch.int)
    # token_type_ids
    token_type_ids = torch.zeros(padded_inputs.shape, dtype=torch.int)
    return padded_inputs, masks, token_type_ids


def collate_fn(batches, is_MLGN):
    if is_MLGN:
        batch_text = []
        batch_label = []
        batch_label_one_hot = []
        for batch in batches:
            batch_text.append(batch[0])
            batch_label_one_hot.append(batch[1])
            batch_label.append(batch[2])


        batch_text_input_ids, text_padding_mask, text_token_type_ids = padding(batch_text, 0)
        batch_label_input_ids, label_padding_mask, label_token_type_ids = padding(batch_label, 0)

        return batch_text_input_ids, text_padding_mask, text_token_type_ids, \
            batch_label_input_ids, label_padding_mask, label_token_type_ids, \
            torch.stack(batch_label_one_hot)
    else:
        batch_text = []
        batch_label_one_hot = []
        for batch in batches:
            batch_text.append(batch[0])
            batch_label_one_hot.append(batch[1])

        batch_text_input_ids, text_padding_mask, text_token_type_ids = padding(batch_text, 0)

        return batch_text_input_ids, text_padding_mask, text_token_type_ids, \
               torch.stack(batch_label_one_hot)

# ! TODO待看
def get_train_data_loader(dataset, tokenizer, is_MLGN, batch_size=16, num_workers=6):
    """
    完全自动：
    aapd：采用auto_split_label_groups
    其他是：
    1) 从 label_dict.pkl 推断 C
    2) 从 processed_train.json 统计 freq (torch.Tensor)
    3) 据此构建 label_groups
    4) 最后再创建真正的 DataLoader 返回（与你原逻辑一致）
    """
    if dataset == 'AAPD':
        train_dataset = AppDataset(dataset, tokenizer, "train", is_MLGN)
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, collate_fn=partial(collate_fn, is_MLGN=is_MLGN))

        return train_data_loader, train_dataset.label_groups
    else:
        # 1&2. 自动统计频次与 C
        freq, C = _count_label_freq_from_json(dataset, split="train")

        # 3. 构建 label_groups（确保传入 torch.Tensor，避免 numpy 的 argsort 类型错误）
        label_groups = make_groups_balanced(C, freq, target_size=64)

        # 4. 构建训练集与 DataLoader（与原逻辑一致）
        train_dataset = AppDataset(dataset, tokenizer, "train", is_MLGN)
        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, is_MLGN=is_MLGN),
        )
        return train_data_loader, label_groups

def get_test_data_loader(dataset, tokenizer, is_MLGN, batch_size=16, num_workers=6):
    test_dataset = AppDataset(dataset, tokenizer, "test", is_MLGN)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, collate_fn=partial(collate_fn, is_MLGN=is_MLGN)
                                  )

    return test_data_loader


def get_label_num(dataset):
    with open(f"./data/{dataset}/label_dict.pkl", "rb") as file:
        label_dict = pickle.load(file)
        return len(label_dict)


# ====== ADDED BLOCK BEGIN: group moe rcv1 ======
def count_label_freq(loader, num_labels, label_key_candidates=('labels','label','y','targets','target'),
                     label_index=-1):
    """
    统计标签频次：
    - 支持 batch 是 dict（从若干候选键取标签）
    - 支持 batch 是 tuple/list（默认取最后一个元素作为标签，或用 label_index 指定）
    - 支持标签是 multi-hot [B, C]、单索引 [B]、或“每样本索引列表”的列表
    """
    freq = torch.zeros(num_labels, dtype=torch.long)

    for batch in loader:
        # 1) 取出标签张量/列表
        if isinstance(batch, dict):
            y = None
            for k in label_key_candidates:
                if k in batch:
                    y = batch[k]
                    break
            if y is None:
                raise KeyError(f"count_label_freq: 找不到标签键，候选={label_key_candidates}")
        elif isinstance(batch, (list, tuple)):
            # 你的 dataloader: (ids, mask, type_ids, label_one_hot) → 默认最后一个是标签
            if not batch:
                continue
            y = batch[label_index]
        else:
            raise TypeError(f"count_label_freq: 不支持的 batch 类型 {type(batch)}")

        # 2) 累加频次
        if torch.is_tensor(y):
            y = y.detach().cpu()
            if y.dim() == 2:
                # multi-hot [B, C]
                if y.size(1) != num_labels:
                    raise ValueError(f"标签维度不匹配: got {y.size(1)} vs num_labels={num_labels}")
                freq += y.long().sum(0)
            elif y.dim() == 1:
                # 单索引 [B]
                freq += torch.bincount(y.long(), minlength=num_labels)
            else:
                raise ValueError(f"不支持的标签张量形状: {tuple(y.shape)} （期望 [B,C] 或 [B]）")
        elif isinstance(y, (list, tuple)):
            # 每个样本一个“标签索引列表”
            for labs in y:
                if torch.is_tensor(labs):
                    labs = labs.detach().cpu().long().tolist()
                freq += torch.bincount(torch.tensor(labs, dtype=torch.long),
                                       minlength=num_labels)
        else:
            raise TypeError(f"不支持的标签类型: {type(y)}")

    return freq

def make_groups_balanced(num_labels, freq, target_size=64):
    """贪心装箱：按频次降序，把标签分配到当前“总频次”最小且尚未满员的组"""
    G = math.ceil(num_labels / target_size)
    groups = [[] for _ in range(G)]
    load = [0] * G
    cap  = [0] * G
    order = torch.argsort(freq, descending=True).tolist()
    for lab in order:
        # 找未满的组里 load 最小的
        cand = [(load[g], g) for g in range(G) if cap[g] < target_size]
        _, g = min(cand)
        groups[g].append(lab)
        load[g] += int(freq[lab])
        cap[g]  += 1
    return groups  # list[list[int]]

def _data_dir_of(dataset_name: str) -> str:
    # 与你当前管线保持一致：./data/{dataset}/...
    return os.path.join(".", "data", dataset_name)

def _load_label_dict(dataset_name: str):
    """读取 label_dict.pkl（label->id 映射）。"""
    path = os.path.join(_data_dir_of(dataset_name), "label_dict.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}，请先用预处理脚本生成。")
    with open(path, "rb") as f:
        label2id = pickle.load(f)
    return label2id  # OrderedDict/Dict[str,int]

def _count_label_freq_from_json(dataset_name: str, split: str = "train"):
    """
    直接从 processed_{split}.json 统计标签频次，返回 (freq_tensor[C], C)。
    好处：不依赖 collate_fn / DataLoader 的 batch 结构，稳。
    """
    data_dir = _data_dir_of(dataset_name)
    json_path = os.path.join(data_dir, f"processed_{split}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到 {json_path}，请确认数据已转换。")

    label2id = _load_label_dict(dataset_name)
    C = len(label2id)
    freq = torch.zeros(C, dtype=torch.long)

    # 读取 JSON 并累计频次（多标签：每个样本的所有标签都+1）
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    # items: [{"text": "...", "label": ["a","b", ...]}, ...]
    for it in items:
        labs = it.get("label", [])
        for lb in labs:
            if lb not in label2id:
                # 严格起见：如果遇到字典中没有的标签，直接报错，避免C与映射不一致
                raise KeyError(f"标签 {lb!r} 不在 label_dict.pkl 映射中，请确认预处理一致性。")
            freq[label2id[lb]] += 1

    return freq, C


class AppDatasetAug(Dataset):
    pass


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('./model/bert-base-uncased', do_lower_case=True)

    train_data_loader = get_test_data_loader(dataset='AAPD', tokenizer=tokenizer, is_MLGN=False)
    # print(label_groups)

    for batch in train_data_loader:
        print(batch)
