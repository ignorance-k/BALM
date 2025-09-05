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
        # 是否使用标签语义
        if self.is_MLGN:
            # eurlex
            # label_encoder = self.tokenizer.encode(" ".join(json_data['label']), add_special_tokens=True)
            # AAPD
            label_text = " ".join([label_dict_change[label] for label in json_data['label']])
            label_encoder = self.tokenizer.encode(label_text + ' ' + json_data['text'], add_special_tokens=True,
                                                  max_length=48, truncation=True)

            return text_encoder, label_one_hot, label_encoder

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


def get_train_data_loader(dataset, tokenizer, is_MLGN, batch_size=16, num_workers=6):
    train_dataset = AppDataset(dataset, tokenizer, "train", is_MLGN)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, collate_fn=partial(collate_fn, is_MLGN=is_MLGN))
    # collate_fn使用lambda有额外参数会导致，序列化报错，也就是num_workers>0, 不行
    # 可以使用partail代替
    # train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
    #                                num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch, is_MLGN))

    return train_data_loader, train_dataset.label_groups


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


class AppDatasetAug(Dataset):
    pass


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('./model/bert-base-uncased', do_lower_case=True)

    train_data_loader = get_test_data_loader(dataset='AAPD', tokenizer=tokenizer, is_MLGN=False)
    # print(label_groups)

    for batch in train_data_loader:
        print(batch)
