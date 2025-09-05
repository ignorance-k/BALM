import os

# 'AAPD', 'EURLex-4k'
dataset = 'AAPD'

# MOE; LORT
# tests = [(False, False), (False, True), (True, False), (True, True)]
use_moe, is_lort = [True, True]
gamma2 = [7, 9, 10]
for loss_name in ['DR', 'ALL']:
    for g in gamma2:
        cmd = f'python main_text_one.py --dataset {dataset} --loss_name {loss_name} --gamma1 {5} --gamma2 {g}'
        if use_moe:
            cmd += " --use_moe"
        if is_lort:
            cmd += " --is_lort"

        bert_version = f"test-9_4-moe-{use_moe} lort-{is_lort}"
        cmd += f' --bert_version "{bert_version}"'

        os.system(cmd)


# for loss_name in ['BCE', 'DR', 'ALL']:
#     cmd = f"python main_text_one.py --dataset {dataset} --loss_name {loss_name}"
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
