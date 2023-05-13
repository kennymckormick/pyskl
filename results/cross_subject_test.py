from __future__ import print_function, absolute_import

import pandas as pd
import numpy as np
import pickle


# Function to calculate top-k accuracy
__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Get the label from XSub_val
data = []
with (open("../data/ntu60_hrnet.pkl", "rb")) as openfile:
    while True:
        try:
            data.append(pickle.load(openfile))
        except EOFError:
            break

# print(data[0].keys())
# print(data[0]['split'].keys())
# print(len(data[0]['split']['xsub_train']))
# print(len(data[0]['split']['xsub_val']))
# print(len(data[0]['split']['xview_train']))
# print(len(data[0]['split']['xview_val']))
# print(data[0]['split']['xsub_val'][0])


# print(len(data[0]['annotations'])) # list annotations
# print(data[0]['annotations'][0].keys())
# print(data[0]['annotations'][0]['label'])


list_xsub_val_names = data[0]['split']['xsub_val']
print(len(list_xsub_val_names))
print(list_xsub_val_names[0])
label = []

print(len(data[0]['annotations']))

for annotation in range(len(data[0]['annotations'])):
    if data[0]['annotations'][annotation]['frame_dir'] in list_xsub_val_names:
        label.append(data[0]['annotations'][annotation]['label'])

print(len(label))
# print(label)

joint_out = []
with (open("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/j_result.pkl", "rb")) as openfile:
    while True:
        try:
            joint_out.append(pickle.load(openfile))
        except EOFError:
            break

bone_out = []
with (open("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/b_result.pkl", "rb")) as openfile:
    while True:
        try:
            bone_out.append(pickle.load(openfile))
        except EOFError:
            break

jm_out = []
with (open("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/jm_result.pkl", "rb")) as openfile:
    while True:
        try:
            jm_out.append(pickle.load(openfile))
        except EOFError:
            break

bm_out = []
with (open("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/bm_result.pkl", "rb")) as openfile:
    while True:
        try:
            bm_out.append(pickle.load(openfile))
        except EOFError:
            break

print("******")
print(len(joint_out))
print("******")
# print(joint_out[0][0])
# print(joint_out[1][0])

joint_out_pd = pd.read_pickle("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/j_result.pkl")
bone_out_pd = pd.read_pickle("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/b_result.pkl")
jm_out_pd = pd.read_pickle("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/jm_result.pkl")
bm_out_pd = pd.read_pickle("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/bm_result.pkl")

# print(len(joint_out_pd))
# print(joint_out_pd[0].shape)
# print(joint_out_pd[0])

action_scores_4s = []
for i in range(len(joint_out_pd)):
    action_scores_4s.append(2 * joint_out_pd[i] + 2 * bone_out_pd[i] + jm_out_pd[i] + bm_out_pd[i])

print(len(action_scores_4s))

# action_scores = 2 * joint_out + 2 * bone_out + jm_out + bm_out

# print(len(action_scores))
action_label_4s = []
for i in range(len(action_scores_4s)):
    action_label_4s.append(np.argmax(action_scores_4s[i]))

print(len(action_label_4s))
# print(type(action_label_4s))

action_scores_2s = []
for i in range(len(joint_out_pd)):
    action_scores_2s.append(joint_out_pd[i] + bone_out_pd[i])

action_label_2s = []
for i in range(len(action_scores_2s)):
    action_label_2s.append(np.argmax(action_scores_2s[i]))

# for i in range(10):
#     print(action_label[i+10])
#     print(label[i+10])
# top1_result = accuracy(action_label, label)
# print(top1_result)

print("Four-stream accuracy Joint : Bone : JM : BM = 2 : 2 : 1 : 1")
accuracy_score_4s = sum(1 for x,y in zip(action_label_4s, label) if x == y) / float(len(action_label_4s))
print(accuracy_score_4s)

print("Two-stream accuracy Joint : Bone = 1 : 1")
accuracy_score_2s = sum(1 for x,y in zip(action_label_2s, label) if x == y) / float(len(action_label_2s))
print(accuracy_score_2s)


action_scores_2s_weight1 = []
for i in range(len(joint_out_pd)):
    action_scores_2s_weight1.append(joint_out_pd[i] + 2 * bone_out_pd[i])

action_label_2s_weight1 = []
for i in range(len(action_scores_2s_weight1)):
    action_label_2s_weight1.append(np.argmax(action_scores_2s_weight1[i]))

accuracy_score_2s_weight1 = sum(1 for x,y in zip(action_label_2s_weight1, label) if x == y) / float(len(action_label_2s_weight1))
print("Two stream accuracy Joint : Bone = 1 : 2")
print(accuracy_score_2s_weight1)

action_scores_2s_weight2 = []
for i in range(len(joint_out_pd)):
    action_scores_2s_weight2.append(joint_out_pd[i] + 3 * bone_out_pd[i])

action_label_2s_weight2 = []
for i in range(len(action_scores_2s_weight2)):
    action_label_2s_weight2.append(np.argmax(action_scores_2s_weight2[i]))

accuracy_score_2s_weight2 = sum(1 for x,y in zip(action_label_2s_weight2, label) if x == y) / float(len(action_label_2s_weight2))
print("Two stream accuracy Joint : Bone = 1 : 3")
print(accuracy_score_2s_weight2)

# action_scores_2s_weight3 = []
# for i in range(len(joint_out_pd)):
#     action_scores_2s_weight3.append(joint_out_pd[i] + 4 * bone_out_pd[i])

# action_label_2s_weight3 = []
# for i in range(len(action_scores_2s_weight3)):
#     action_label_2s_weight1.append(np.argmax(action_scores_2s_weight3[i]))

# accuracy_score_2s_weight3 = sum(1 for x,y in zip(action_label_2s_weight3, label) if x == y) / float(len(action_label_2s_weight3))
# print("Two stream accuracy Joint : Bone = 1 : 4")
# print(accuracy_score_2s_weight3)

