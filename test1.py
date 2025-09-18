# save as: sanity_check_labels.py
import json, pickle, os, collections

DATASET = "Wiki"  # 改成你的数据集名，比如 "AAPD"
root = os.path.join("./data", DATASET)
js = os.path.join(root, "processed_train.json")
lbl = os.path.join(root, "label_dict.pkl")

with open(js, "r", encoding="utf-8") as f: items = json.load(f)
with open(lbl, "rb") as f: label2id = pickle.load(f)

n = len(items)
empty = sum(1 for x in items if not x.get("label"))
all_labels = [lb for x in items for lb in x.get("label", [])]
ctr = collections.Counter(all_labels)

print(f"样本数={n}")
print(f"无标签样本={empty} ({empty*100.0/max(n,1):.2f}%)")
print(f"label_dict标签数={len(label2id)}（前10个：{list(label2id)[:10]}）")
print("前10个最频繁标签：", ctr.most_common(10))
print("总标签出现次数=", sum(ctr.values()))
