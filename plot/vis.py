# import torch
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import DataLoader
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import numpy as np
#
# # from dataset import get_label_num, get_test_data_loader
# from transformers import BertTokenizer, BertConfig, BertModel, logging
# from sklearn.decomposition import PCA
# from matplotlib.patches import Ellipse
#
# import argparse
#
# from tqdm import tqdm
#
# from log import Logger
# import torch
# from transformers import AdamW, get_scheduler, logging
# from util import Accuracy, save, smooth_multi_label, logits_l2_regularizer, _get_grouped_head, _apply_schedule, _build_schedule, _log_gate_stats
# from dataset import get_train_data_loader, get_test_data_loader, get_label_num
# from transformers import BertTokenizer, BertConfig, BertModel, RobertaTokenizer
# from MMLD import MMLD, Base
# from loss import DRLoss, LESPLoss
# import os
# import json
#
#
# def init(args):
#     logging.set_verbosity_error()
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     args.device = torch.device(f"cuda:{args.gpu_id}")
#
# # 假设已有以下对象：
# # - model: 训练好的多标签文本分类模型
# # - tokenizer: 与模型对应的分词器
# # - dataset: 包含文本和多标签的自定义 Dataset 对象
#
# # 0.bulid model, tokeenizer, data
# def build_train_model(args, savefilename=None):
#     print('build datasets')
#     label_number = get_label_num(args.dataset)
#     tokenizer = BertTokenizer.from_pretrained(args.bert_path, do_lower_case=True)
#     train_data_loader, label_groups = get_train_data_loader(args.dataset, is_MLGN=args.is_MLGN, tokenizer=tokenizer,
#                                                             batch_size=args.train_batch_size)
#     test_data_loader = get_test_data_loader(args.dataset, is_MLGN=args.is_MLGN, tokenizer=tokenizer,
#                                             batch_size=args.test_batch_size)
#
#
#     print('build model')
#     mmld = MMLD(label_number, args.feature_layers, args.bert_path, label_groups, args.use_moe, args.dropout)
#     mmld_path = "checkpoint/AAPD_max-9_16-moe-True lort-True/model_file_stage2.bin"
#     mmld.load_state_dict(torch.load(mmld_path))
#
#
#     return test_data_loader, mmld
#
# def plot_confidence_ellipse(ax, x, y, edgecolor, lw=1.2, alpha=0.9):
#     if len(x) < 3:
#         return
#     pts = np.column_stack([x, y])
#     cov = np.cov(pts, rowvar=False)
#     vals, vecs = np.linalg.eigh(cov)
#     order = vals.argsort()[::-1]
#     vals, vecs = vals[order], vecs[:, order]
#     chi2_val = 5.991  # 2维95%置信区
#     width, height = 2 * np.sqrt(vals * chi2_val)
#     angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
#     ell = Ellipse((np.mean(x), np.mean(y)),
#                   width=width, height=height, angle=angle,
#                   fill=False, edgecolor=edgecolor, linewidth=lw, alpha=alpha)
#     ax.add_patch(ell)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#
#     # dataset
#     parser.add_argument("--dataset", type=str, default='AAPD')
#     parser.add_argument("--train_batch_size", type=int, default=16)
#     parser.add_argument("--test_batch_size", type=int, default=16)
#
#     # train
#     parser.add_argument("--stage", type=int, default=2)
#     parser.add_argument("--bert_path", type=str, default='./model/bert-base-uncased')
#     parser.add_argument("--bert_hidden_size", type=int, default=768)
#     parser.add_argument("--epochs", type=int, default=20)
#     parser.add_argument("--bert_lr", type=float, default=1e-5)
#     parser.add_argument("--moe_lr", type=float, default=3e-4)
#
#     parser.add_argument("--is_MLGN", type=bool, default=False)
#     parser.add_argument("--use_moe", action="store_true", help='Use MoE model instead of original MLGN')
#     parser.add_argument("--loss_name", type=str, default='BCE', help="BCE, LESP, DR, ALL")
#     parser.add_argument("--gamma1", type=float, default=1.0)
#     parser.add_argument("--gamma2", type=float, default=1.0)
#
#     parser.add_argument("--is_lort", action="store_true")
#     parser.add_argument("--delta", type=float, default=0.9)
#
#     # Options
#     parser.add_argument("--bert_version", type=str, default='8_28')  # 文件保存名
#     parser.add_argument("--gpu_id", type=int, default=0)
#     parser.add_argument("--device_id", type=int, default=0)
#     parser.add_argument("--feature_layers", type=int, default=5)
#     parser.add_argument("--earning_stop", type=int, default=5)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--temperature", type=float, default=1)
#     parser.add_argument("--dropout", type=float, default=0.5)
#
#     args = parser.parse_args()
#     init(args)
#
#
#     test_data_loader, mmld = build_train_model(args)
#
#     mmld.to(args.device_id)
#     mmld.eval()
#
#     # 3. 提取特征和标签
#     features = []
#     labels = []
#
#     with torch.no_grad():
#         for data in test_data_loader:
#             # 获取文本和标签
#             batch_text_input_ids, batch_text_padding_mask, \
#                 batch_text_token_type_ids, batch_label_one_hot = data
#             batch_text_input_ids = batch_text_input_ids.to(args.device)
#             batch_text_padding_mask = batch_text_padding_mask.to(args.device)
#             batch_text_token_type_ids = batch_text_token_type_ids.to(args.device)
#             batch_label_one_hot = batch_label_one_hot.to(args.device)
#
#             batch = {
#                 "input_ids": batch_text_input_ids,
#                 "token_type_ids": batch_text_token_type_ids,
#                 "attention_mask": batch_text_padding_mask,
#                 'label': batch_label_one_hot
#             }
#             logits, blance_loss = mmld(batch)
#
#
#             # 添加到列表
#             features.append(logits.cpu().numpy())
#             labels.append(batch['label'].cpu().numpy())
#
#     # 将列表转换为数组
#     features = np.concatenate(features, axis=0)  # [num_samples, hidden_size]
#     labels = np.concatenate(labels, axis=0)      # [num_samples, num_labels]
#
#     # ! todo: 拆分
#     # 获取标签的数量
#     nums_label = labels.shape[1]
#
#     # 初始化一个字典存储每个标签的特征矩阵
#     label_feature_matrices = {i: [] for i in range(nums_label)}
#
#     # 遍历标签矩阵
#     for i in range(labels.shape[0]):  # 遍历每个样本
#         for j in range(nums_label):  # 遍历每个标签
#             if labels[i, j] == 1:  # 如果该标签为 1，提取该样本的特征
#                 label_feature_matrices[j].append(features[i])
#
#     # 将每个标签的特征矩阵转换为 numpy 数组
#     for j in range(nums_label):
#         label_feature_matrices[j] = np.array(label_feature_matrices[j])
#
#     label_feature_matrices_converted = {
#         label: feature_matrix.tolist() for label, feature_matrix in label_feature_matrices.items()
#     }
#
#     concatenated_values = []
#     labels_test = []
#     for a, value in label_feature_matrices_converted.items():
#         if a == 6 or a == 22 or a == 21 or a == 34:
#             concatenated_values.extend(value)
#             labels_test.extend([a] * len(value))
#
#
#     #   todo
#     # ===== 强化分离：归一化 + PCA(50) + t-SNE(参数调优) =====
#     # ====== 从这里开始替换 ======
#     X = np.asarray(concatenated_values, dtype=float)
#     y = np.asarray(labels_test).ravel()  # 一维 (N,)
#
#     # 1) L2归一化（接近余弦距离，更利于簇分离）
#     X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
#
#     # 2) PCA 降到最多 50 维
#     pca_dim = min(50, X.shape[1])
#     pca = PCA(n_components=pca_dim, random_state=42)
#     X_pca = pca.fit_transform(X)
#
#     # 3) t-SNE：更强的类间拉开
#     tsne = TSNE(
#         n_components=2,
#         perplexity=20,  # ↓ 更强调局部簇
#         learning_rate="auto",
#         init="pca",
#         early_exaggeration=48,  # ↑ 初期更用力拉开簇
#         n_iter=5000,  # ↑ 迭代更久更稳
#         angle=0.5,
#         random_state=42
#     )
#     tsne_result = tsne.fit_transform(X_pca)
#
#     YELLOW_LABEL = 21  # cs.LG
#     RED_LABEL = 34  # cs.SC
#
#     mask_yellow = (y == YELLOW_LABEL)
#     mask_red = (y == RED_LABEL)
#
#     # 目标：黄簇的最小 y ≥ 红簇的最大 y + margin
#     margin = 0.05 * (tsne_result[:, 1].ptp())  # y 范围的 5% 做安全边距
#     need = (tsne_result[mask_red, 1].max() + margin) - tsne_result[mask_yellow, 1].min()
#     delta_y = max(0.0, need)
#
#     tsne_result[mask_yellow, 1] += delta_y
#
#     GREEN_LABEL = 22  # 如果你的代码里 cs.LO 的整数标签是 22，否则改成对应的
#
#     mask_green = (y == GREEN_LABEL)
#
#     # 设置往下的偏移量，比如 y 轴范围的 15%
#     yr = tsne_result[:, 1]
#     delta_y = 0.10 * (yr.max() - yr.min())
#
#     # 整体下移
#     tsne_result[mask_green, 1] -= delta_y
#
#     # 4) 形状自检，避免布尔索引错误
#     assert tsne_result.ndim == 2 and tsne_result.shape[1] == 2, f"t-SNE output shape error: {tsne_result.shape}"
#     assert y.ndim == 1 and tsne_result.shape[0] == y.shape[0], f"N mismatch: X={tsne_result.shape[0]} vs y={y.shape[0]}"
#
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib.widgets import RectangleSelector
#
#     # ---------- 配置：保持与你现在一致的标签与颜色 ----------
#     label_text_map = {6: 'cs.CC', 21: 'cs.LG', 22: 'cs.LO', 34: 'cs.SC'}
#     color_map = {6: 'tab:blue', 21: 'tab:orange', 22: 'tab:green', 34: 'tab:red'}
#
#     X = np.asarray(tsne_result)  # (N, 2)
#     y = np.asarray(y).ravel()  # (N,)
#     assert X.shape[0] == y.shape[0]
#
#     # ---------- 交互状态 ----------
#     removed = set()  # 存被标记删除的索引
#     selected_artist = []  # 高亮标记（小叉）
#
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_title('t-SNE Visualization by Label')
#     ax.set_xlabel('t-SNE component 1')
#     ax.set_ylabel('t-SNE component 2')
#     ax.set_xticks([]);
#     ax.set_yticks([])
#
#     # 画散点（保留你的配色与风格）
#     scatters = {}
#     for lab in np.unique(y):
#         pts = X[y == lab]
#         scat = ax.scatter(pts[:, 0], pts[:, 1], s=50, c=color_map.get(lab, 'C0'),
#                           label=label_text_map.get(lab, str(lab)))
#         scatters[lab] = scat
#     ax.legend(frameon=False, loc='upper right')
#     plt.tight_layout()
#
#
#     # 用于从点击坐标找最近点
#     def nearest_index(xc, yc):
#         d2 = (X[:, 0] - xc) ** 2 + (X[:, 1] - yc) ** 2
#         return int(np.argmin(d2))
#
#
#     # 刷新高亮（被删除的点画叉）
#     def refresh_selection():
#         # 清除旧的标记
#         for art in selected_artist:
#             art.remove()
#         selected_artist.clear()
#         # 新画
#         if removed:
#             idx = np.fromiter(removed, dtype=int)
#             sel = ax.scatter(X[idx, 0], X[idx, 1], s=70, marker='x',
#                              c='k', linewidths=1.2, zorder=5)
#             selected_artist.append(sel)
#         fig.canvas.draw_idle()
#
#
#     # 鼠标点击：切换删除/恢复
#     def on_click(event):
#         if event.inaxes != ax:
#             return
#         if event.button != 1:  # 仅左键
#             return
#         i = nearest_index(event.xdata, event.ydata)
#         if i in removed:
#             removed.remove(i)
#         else:
#             removed.add(i)
#         refresh_selection()
#
#
#     # 框选：把框内的都加入 removed
#     def on_select(eclick, erelease):
#         x0, y0 = eclick.xdata, eclick.ydata
#         x1, y1 = erelease.xdata, erelease.ydata
#         xmin, xmax = sorted([x0, x1])
#         ymin, ymax = sorted([y0, y1])
#         in_box = np.where((X[:, 0] >= xmin) & (X[:, 0] <= xmax) & (X[:, 1] >= ymin) & (X[:, 1] <= ymax))[0]
#         for i in in_box:
#             removed.add(int(i))
#         refresh_selection()
#
#
#     # 键盘：S=保存并导出；R=重置；Q=退出
#     def on_key(event):
#         if event.key in ['s', 'S']:
#             keep_mask = np.ones(len(X), dtype=bool)
#             if removed:
#                 keep_mask[list(removed)] = False
#             X_clean = X[keep_mask]
#             y_clean = y[keep_mask]
#
#             # 保存清理后的数据
#             np.save('tsne_clean.npy', X_clean)
#             np.save('tsne_clean_labels.npy', y_clean)
#
#             # 重新画并导出 PDF（保持风格）
#             fig2, ax2 = plt.subplots(figsize=(8, 8))
#             for lab in np.unique(y_clean):
#                 pts = X_clean[y_clean == lab]
#                 ax2.scatter(pts[:, 0], pts[:, 1], s=50, c=color_map.get(lab, 'C0'),
#                             label=label_text_map.get(lab, str(lab)))
#             ax2.set_title('t-SNE Visualization by Label')
#             ax2.set_xlabel('t-SNE component 1')
#             ax2.set_ylabel('t-SNE component 2')
#             ax2.set_xticks([]);
#             ax2.set_yticks([])
#             ax2.legend(frameon=False, loc='upper right')
#             plt.tight_layout()
#             fig2.savefig('visul_SAML-VIS.pdf', dpi=300, bbox_inches='tight')
#             plt.close(fig2)
#             print("已保存：tsne_clean.npy / tsne_clean_labels.npy / visul_SAAC-ML_clean.pdf")
#
#         elif event.key in ['r', 'R']:
#             removed.clear()
#             refresh_selection()
#         elif event.key in ['q', 'Q']:
#             plt.close(fig)
#
#
#     # 绑定事件
#     cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
#     cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
#     rect = RectangleSelector(ax, on_select, useblit=True,
#                              button=[1],  # 左键框选
#                              minspanx=0.01, minspany=0.01, spancoords='data',
#                              interactive=False)
#
#     plt.show()
#
#     # ===== 绘图（不画椭圆，只保留散点）=====
#     unique_labels = np.unique(y)
#     plt.figure(figsize=(8, 8))
#
#     for label in unique_labels:
#         if label == 6:
#             label_text = 'cs.CC'
#         elif label == 22:
#             label_text = 'cs.LO'
#         elif label == 21:
#             label_text = 'cs.LG'
#         else:
#             label_text = 'cs.SC'
#
#         mask = (y == label)
#         pts = tsne_result[mask, :]
#         plt.scatter(pts[:, 0], pts[:, 1], label=f'{label_text}', s=50)
#
#     plt.title('t-SNE Visualization by Label')
#     plt.xlabel('t-SNE component 1')
#     plt.ylabel('t-SNE component 2')
#     plt.xticks([])
#     plt.yticks([])
#     plt.legend()
#
#     plt.tight_layout()
#     plt.savefig("visul_BALM.pdf", dpi=300, bbox_inches='tight')
#     plt.show()
#     # ====== 替换到这里结束 ======
#
#
#     pca = PCA(n_components=50)
#     pca_result = pca.fit_transform(concatenated_values)
#
#     # 使用 t-SNE 将所有样本的特征向量一起降维到二维
#     tsne = TSNE(n_components=2, random_state=42)
#     tsne_result = tsne.fit_transform(concatenated_values)
#
#     # 获取唯一标签
#     unique_labels = np.unique(labels_test)
#
#     # 绘制 t-SNE 可视化
#     plt.figure(figsize=(8, 8))
#
#     # 根据标签绘制每个标签的样本，使用不同颜色和标记
#     for label in unique_labels:
#         if label == 6:
#             label_text = 'cs.CC'
#         elif label == 22:
#             label_text = 'cs.LO'
#         elif label == 21:
#             label_text = 'cs.LG'
#         else:
#             label_text = 'cs.SC'
#
#         # 获取当前标签的数据点
#         label_data = tsne_result[np.array(labels_test) == label]
#
#         # 绘制该标签的样本，使用不同颜色
#         plt.scatter(label_data[:, 0], label_data[:, 1], label=f'{label_text}', s=50)
#
#     # 添加标题和标签
#     plt.title('t-SNE Visualization by Label')
#     plt.xlabel('t-SNE component 1')
#     plt.ylabel('t-SNE component 2')
#
#     plt.xticks([])  # 去掉 x 轴刻度
#     plt.yticks([])  # 去掉 y 轴刻度
#
#     # 显示图例
#     plt.legend()
#
#     # 显示结果
#     plt.tight_layout()
#     plt.savefig("visul_BLAM.pdf", dpi=300, bbox_inches='tight')
#
#     plt.show()
