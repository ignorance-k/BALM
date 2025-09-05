import argparse

from tqdm import tqdm

from log import Logger
import torch
from transformers import AdamW, get_scheduler, logging
from util import Accuracy, save, smooth_multi_label, logits_l2_regularizer
from dataset import get_train_data_loader, get_test_data_loader, get_label_num
from transformers import BertTokenizer, BertConfig, BertModel, RobertaTokenizer
from MMLD import MMLD, Base
from loss import DRLoss, LESPLoss
import os
import json

def init(args):
    logging.set_verbosity_error()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device(f"cuda:{args.gpu_id}")


def build_train_model_stage1(args):
    print('training stage 1')
    print('build data loader')
    label_number = get_label_num(args.dataset)
    tokenizer = BertTokenizer.from_pretrained(args.bert_path, do_lower_case=True)
    train_data_loader, label_groups = get_train_data_loader(args.dataset, is_MLGN=args.is_MLGN, tokenizer=tokenizer,
                                                            batch_size=args.train_batch_size)
    test_data_loader = get_test_data_loader(args.dataset, is_MLGN=args.is_MLGN, tokenizer=tokenizer,
                                            batch_size=args.test_batch_size)

    print('build model')
    model_stage1 = Base(label_number, args.bert_path, args.feature_layers, args.dropout)

    print('build loss')
    BCE_loss = torch.nn.BCEWithLogitsLoss()

    print("build optimizer")
    model_stage1_train_parameters = [parameter for parameter in model_stage1.parameters() if parameter.requires_grad]
    optimizer1 = AdamW(model_stage1_train_parameters, lr=args.lr1)

    print("build lr_scheduler")
    lr_scheduler1 = get_scheduler("linear", optimizer=optimizer1, num_warmup_steps=0,
                                 num_training_steps=args.epochs_stage1 * len(train_data_loader))
    return train_data_loader, test_data_loader, model_stage1, BCE_loss, optimizer1, lr_scheduler1

def build_train_model(args, savefilename=None):
    print('build datasets')
    label_number = get_label_num(args.dataset)
    tokenizer = BertTokenizer.from_pretrained(args.bert_path, do_lower_case=True)
    train_data_loader, label_groups = get_train_data_loader(args.dataset, is_MLGN=args.is_MLGN, tokenizer=tokenizer,
                                                            batch_size=args.train_batch_size)
    test_data_loader = get_test_data_loader(args.dataset, is_MLGN=args.is_MLGN, tokenizer=tokenizer,
                                            batch_size=args.test_batch_size)


    print('build model')
    mmld = MMLD(label_number, args.feature_layers, args.bert_path, label_groups, args.use_moe, args.dropout)


    print('build loss')
    BCE_loss = torch.nn.BCEWithLogitsLoss()
    LESP_loss = LESPLoss()
    DR_loss = DRLoss(args.gamma1, args.gamma2)

    print("build optimizer")
    # MOE的分段学习率
    bert_params, head_params = [], []
    for n, p in mmld.named_parameters():
        if n.startswith('bert.'):
            bert_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': 1e-5, 'weight_decay': 0.01},
        {'params': head_params, 'lr': 3e-4, 'weight_decay': 0.01},
    ])

    # mmld_train_parameters = [parameter for parameter in mmld.parameters() if parameter.requires_grad]
    # optimizer = AdamW(mmld_train_parameters, lr=args.lr)

    print("build lr_scheduler")
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=args.epochs * len(train_data_loader))

    return train_data_loader, test_data_loader, mmld, BCE_loss, LESP_loss, DR_loss, optimizer, lr_scheduler, label_number

def run_train_batch(args, data, model, BCE_loss, optimizer, lr_scheduler, label_number=None, LESP_loss=None, DR_loss=None):
    # deivice
    batch_text_input_ids, batch_text_padding_mask,  \
                            batch_text_token_type_ids, batch_label_one_hot = data
    batch_text_input_ids = batch_text_input_ids.to(args.device)
    batch_text_padding_mask = batch_text_padding_mask.to(args.device)
    batch_text_token_type_ids = batch_text_token_type_ids.to(args.device)
    batch_label_one_hot = batch_label_one_hot.to(args.device)

    model = model.to(args.device)

    batch = {
        "input_ids": batch_text_input_ids,
        "token_type_ids": batch_text_token_type_ids,
        "attention_mask": batch_text_padding_mask,
        'label': batch_label_one_hot
    }

    logits, blance_loss = model(batch)

    # if args.stage == 1:
    #     criterion = BCE_loss(logits, batch['label'])
    #
    #     optimizer.zero_grad()
    #     criterion.backward()
    #     optimizer.step()
    #     lr_scheduler.step()
    #
    #     return criterion

    # if args.stage == 2:
    #
    # if args.is_lort == True:
    #     smoothed_labels = smooth_multi_label(batch['label'], label_number, args.delta)
    # else:
    #     smoothed_labels = batch['label']

    if args.loss_name == 'BCE':
        criterion = BCE_loss(logits, batch['label'])
    elif args.loss_name == 'LESP':
        criterion = LESP_loss(logits, batch['label'])
    elif args.loss_name == 'DR':
        criterion = DR_loss(logits, batch['label'])
    elif args.loss_name == 'ALL':
        criterion = 0.5 * BCE_loss(logits, batch['label']) + 0.5 * DR_loss(logits, batch['label'])

    if args.is_lort:
        loss_reg = logits_l2_regularizer(logits)
    else:
        loss_reg = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    loss = criterion + loss_reg + blance_loss

    optimizer.zero_grad()
    loss.backward()
        
    optimizer.step()
    lr_scheduler.step()

    return loss

def run_test_batch(args, data, model, BCE_loss, accuracy, LESP_loss=None, DR_loss=None):
    batch_text_input_ids, batch_text_padding_mask, \
                                batch_text_token_type_ids, batch_label_one_hot = data
    batch_text_input_ids = batch_text_input_ids.to(args.device)
    batch_text_padding_mask = batch_text_padding_mask.to(args.device)
    batch_text_token_type_ids = batch_text_token_type_ids.to(args.device)
    batch_label_one_hot = batch_label_one_hot.to(args.device)

    model = model.to(args.device)

    batch = {
        "input_ids": batch_text_input_ids,
        "token_type_ids": batch_text_token_type_ids,
        "attention_mask": batch_text_padding_mask,
        'label': batch_label_one_hot
    }

    logits, blance_loss = model(batch)
    accuracy.calc(logits, batch['label'])
    #

    # if args.stage == 1:
    #     criterion = BCE_loss(logits, batch['label'])
    #     return criterion

    # if args.stage == 2:

    if args.loss_name == 'BCE':
        criterion = BCE_loss(logits, batch['label'])
    elif args.loss_name == 'LESP':
        criterion = LESP_loss(logits, batch['label'])
    elif args.loss_name == 'DR':
        criterion = DR_loss(logits, batch['label'])
    elif args.loss_name == 'ALL':
        criterion = 0.5 * BCE_loss(logits, batch['label']) + 0.5 * DR_loss(logits, batch['label'])

    return criterion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset", type=str, default='AAPD')
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)

    # train
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--bert_path", type=str, default='./model/bert-base-uncased')
    parser.add_argument("--bert_hidden_size", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--is_MLGN", type=bool, default=False)
    parser.add_argument("--use_moe", action="store_true", help='Use MoE model instead of original MLGN')
    parser.add_argument("--loss_name", type=str, default='BCE', help="BCE, LESP, DR, ALL")
    parser.add_argument("--gamma1", type=float, default=1.0)
    parser.add_argument("--gamma2", type=float, default=1.0)

    parser.add_argument("--is_lort", action="store_true")
    parser.add_argument("--delta", type=float, default=0.9)


    # Options
    parser.add_argument("--bert_version", type=str, default='8_28') # 文件保存名
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--feature_layers", type=int, default=5)
    parser.add_argument("--earning_stop", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)

    args = parser.parse_args()

    # args.timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    save_filename = args.dataset + '_' + args.bert_version

    save_dir = f"./checkpoint/{save_filename}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    args_dict = vars(args)
    with open(os.path.join(save_dir, 'argument.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)


    init(args)
    print(args)     # use_moe, loss_name, lort

    accuracy = Accuracy()
    train_data_loader, test_data_loader, mmld, BCE_loss, LESP_loss, DR_loss, optimizer, lr_scheduler, label_number = build_train_model(
        args)

    save_acc1, save_acc3, save_acc5 = 0, 0, 0
    max_only_d5 = 0
    num_stop_dropping = 0

    for epoch in range(args.epochs):
        mmld.train()
        # 把train_data_looader封装为一个进度条对象，自动跟踪管理
        # batch.set_xxx，自定义进度条
        # torch.cuda.empty_cache()
        with tqdm(train_data_loader, ncols=200) as batch:
            for batch_idx, data in enumerate(batch):
                criterion = run_train_batch(args, data, mmld, BCE_loss, optimizer, lr_scheduler, label_number,
                                            LESP_loss, DR_loss)
                batch.set_description(f"train epoch:{epoch + 1}/{args.epochs}")
                batch.set_postfix(loss=criterion.item())

        with torch.no_grad():
            mmld.eval()
            accuracy.reset_acc()
            with tqdm(test_data_loader, ncols=200) as batch:
                for data in batch:
                    _loss = run_test_batch(args, data, mmld, BCE_loss, accuracy, LESP_loss, DR_loss)
                    batch.set_description(f"stage1 test epoch:{epoch + 1}/{args.epochs}")
                    loss = _loss.item()
                    p1 = accuracy.get_acc1()
                    p3 = accuracy.get_acc3()
                    p5 = accuracy.get_acc5()
                    d3 = accuracy.get_ndcg3()
                    d5 = accuracy.get_ndcg5()
                    batch.set_postfix(loss=loss, p1=p1, p3=p3, p5=p5, d3=d3, d5=d5)

        log_str = f'stage1-{epoch:>2}: {p1:.4f}, {p3:.4f}, {p5:.4f}, {d3:.4f}, {d5:.4f}, train_loss:{criterion.item()}'
        LOG = Logger(save_filename)
        LOG.log(log_str)

        if max_only_d5 < d5:
            num_stop_dropping = 0
            max_only_d5 = d5
            save(args, save_dir, log_str, mmld)
        else:
            num_stop_dropping += 1

        # earning stop
        if num_stop_dropping >= args.earning_stop:
            print('Have not increased for %d check points, early stop training' % num_stop_dropping)
            break

    # args.stage = 2
    #
    #
    # if args.stage == 2:
    #     '''stage2'''
    #     # 加载组件
    #     accuracy = Accuracy()
    #     train_data_loader, test_data_loader, mmld, BCE_loss, LESP_loss, DR_loss, optimizer2, lr_scheduler2, label_number = build_train_model(args, save_dir)
    #
    #     save_acc1, save_acc3, save_acc5 = 0, 0, 0
    #     max_only_d5 = 0
    #     num_stop_dropping = 0
    #
    #     for epoch in range(args.epochs_stage2):
    #         mmld.train()
    #         # 把train_data_looader封装为一个进度条对象，自动跟踪管理
    #         # batch.set_xxx，自定义进度条
    #         # torch.cuda.empty_cache()
    #         with tqdm(train_data_loader, ncols=200) as batch:
    #             for batch_idx, data in enumerate(batch):
    #                 criterion = run_train_batch(args, data, mmld, BCE_loss, optimizer2, lr_scheduler2, label_number, LESP_loss, DR_loss)
    #                 batch.set_description(f"stage2 train epoch:{epoch + 1}/{args.epochs_stage2}")
    #                 batch.set_postfix(loss=criterion.item())
    #
    #         with torch.no_grad():
    #             mmld.eval()
    #             accuracy.reset_acc()
    #             with tqdm(test_data_loader, ncols=200) as batch:
    #                 for data in batch:
    #                     _loss = run_test_batch(args, data, mmld, BCE_loss, accuracy, LESP_loss, DR_loss)
    #                     batch.set_description(f"stage2 test epoch:{epoch + 1}/{args.epochs_stage2}")
    #                     loss = _loss.item()
    #                     p1 = accuracy.get_acc1()
    #                     p3 = accuracy.get_acc3()
    #                     p5 = accuracy.get_acc5()
    #                     d3 = accuracy.get_ndcg3()
    #                     d5 = accuracy.get_ndcg5()
    #                     batch.set_postfix(loss=loss, p1=p1, p3=p3, p5=p5, d3=d3, d5=d5)
    #
    #         log_str = f'stage-2 {epoch:>2}: {p1:.4f}, {p3:.4f}, {p5:.4f}, {d3:.4f}, {d5:.4f}, train_loss:{criterion.item()}'
    #         LOG = Logger(save_filename)
    #         LOG.log(log_str)
    #
    #         if max_only_d5 < d5:
    #             num_stop_dropping = 0
    #             max_only_d5 = d5
    #             save(args, save_dir, log_str, mmld)
    #         else:
    #             num_stop_dropping += 1
    #
    #         # earning stop
    #         if num_stop_dropping >= args.earning_stop:
    #             print('Have not increased for %d check points, early stop training' % num_stop_dropping)
    #             break