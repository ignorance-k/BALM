import os

dataset = 'AAPD'
use_moe, is_lort = True, True

cmd = f"python main_text_one.py --dataset {dataset} --loss_name {'BCE'}"

if use_moe:
    cmd += " --use_moe"
if is_lort:
    cmd += " --is_lort"

bert_version = f"max-9_16-moe-{use_moe} lort-{is_lort}"
cmd += f' --bert_version "{bert_version}"'

os.system(cmd)


# 'AAPD', 'EURLex-4k'
# dataset = 'rcv1'
# tests = [(False, False), (True, False), (True, True)]
# for use_moe, is_lort in tests:
#     cmd = f"python main_text_one.py --dataset {dataset} --loss_name {'BCE'} --epochs {20} --bert_lr {1e-5} \
#             --train_batch_size {32} --seed {19950919}"
#
#     if use_moe:
#         cmd += " --use_moe"
#     if is_lort:
#         cmd += " --is_lort"
#
#     bert_version = f"test-9_12-moe-{use_moe} lort-{is_lort}"
#     cmd += f' --bert_version "{bert_version}"'
#
#     os.system(cmd)


#
# for use_moe, is_lort in tests:
#     cmd = f"python main_text_one.py --dataset {dataset} --loss_name {ALL}"
#
#     if use_moe:
#         cmd += " --use_moe"
#     if is_lort:
#         cmd += " --is_lort"
#
#     bert_version = f"test-9_3-moe-{use_moe} lort-{is_lort}"
#     cmd += f' --bert_version "{bert_version}"'
#
#     os.system(cmd)


#
# command = f'python main_text_one.py --dataset {dataset} --use_moe --is_lort --loss_name {"DR"} --gamma1 {g1} --gamma2 {g2} \
#              --bert_version {"test-" + "8_28_g7"}'


# os.system(command)




# for use_moe, is_lort in tests:
#     command = f'python main_text.py --dataset {dataset} --use_moe {use_moe} --is_lort {is_lort} \
#             --bert_version {"test-" + "moe:" + str(use_moe) + " lort:" + str(is_lort)}'
#     os.system(command)



# for g1 in gamma1:
#     for g2 in gamma2:
#         command = f'python main_MLGN.py --dataset {dataset} --loss_name {loss_name} --gamma1 {g1} --gamma2 {g2} \
#         --bert_version { "moe-" + loss_name}'
#         os.system(command)
