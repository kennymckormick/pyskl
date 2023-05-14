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

# Read pkl file with pickle
with (open("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/j_result.pkl", "rb")) as openfile:
    joint_out = pickle.load(openfile)

with (open("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/b_result.pkl", "rb")) as openfile:
    bone_out = pickle.load(openfile)

jm_out = []
with (open("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/jm_result.pkl", "rb")) as openfile:
    jm_out = pickle.load(openfile)

with (open("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/bm_result.pkl", "rb")) as openfile:
    bm_out = pickle.load(openfile)

print("******")
print(len(joint_out))
print("******")
# print(joint_out[0][0])
# print(joint_out[1][0])

# Read pkl file with pandas
joint_out_pd = pd.read_pickle("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/j_result.pkl")
bone_out_pd = pd.read_pickle("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/b_result.pkl")
jm_out_pd = pd.read_pickle("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/jm_result.pkl")
bm_out_pd = pd.read_pickle("stgcn/stgcn_pyskl_ntu60_xsub_hrnet/bm_result.pkl")

# print(len(joint_out_pd))
# print(joint_out_pd[0].shape)
# print(joint_out_pd[0])
# print(type(action_label_4s))

### Cross Subject Two stream experiments - (Joint : Bone)
# Two stream: Joint : Bone = 1 : 1
action_scores_2s = []
for i in range(len(joint_out_pd)):
    action_scores_2s.append(joint_out_pd[i] + bone_out_pd[i])

action_label_2s = []
for i in range(len(action_scores_2s)):
    action_label_2s.append(np.argmax(action_scores_2s[i]))

print("Two-stream accuracy Joint : Bone = 1 : 1")
accuracy_score_2s = sum(1 for x,y in zip(action_label_2s, label) if x == y) / float(len(action_label_2s))
print(accuracy_score_2s)

# Two stream: Joint : Bone = 1 : 2
action_scores_2s_weight2 = []
for i in range(len(joint_out_pd)):
    action_scores_2s_weight2.append(joint_out_pd[i] + 2 * bone_out_pd[i])

action_label_2s_weight2 = []
for i in range(len(action_scores_2s_weight2)):
    action_label_2s_weight2.append(np.argmax(action_scores_2s_weight2[i]))

accuracy_score_2s_weight2 = sum(1 for x,y in zip(action_label_2s_weight2, label) if x == y) / float(len(action_label_2s_weight2))
print("Two stream accuracy Joint : Bone = 1 : 2")
print(accuracy_score_2s_weight2)

# Two stream: Joint : Bone = 1 : 3
action_scores_2s_weight3 = []
for i in range(len(joint_out_pd)):
    action_scores_2s_weight3.append(joint_out_pd[i] + 3 * bone_out_pd[i])

action_label_2s_weight3 = []
for i in range(len(action_scores_2s_weight3)):
    action_label_2s_weight3.append(np.argmax(action_scores_2s_weight3[i]))

accuracy_score_2s_weight3 = sum(1 for x,y in zip(action_label_2s_weight3, label) if x == y) / float(len(action_label_2s_weight3))
print("Two stream accuracy Joint : Bone = 1 : 3")
print(accuracy_score_2s_weight3)

# Two stream: Joint : Bone = 1 : 4
action_scores_2s_weight4 = []
for i in range(len(joint_out_pd)):
    action_scores_2s_weight4.append(joint_out_pd[i] + 4 * bone_out_pd[i])

action_label_2s_weight4 = []
for i in range(len(action_scores_2s_weight4)):
    action_label_2s_weight4.append(np.argmax(action_scores_2s_weight4[i]))

accuracy_score_2s_weight4 = sum(1 for x,y in zip(action_label_2s_weight4, label) if x == y) / float(len(action_label_2s_weight4))
print("Two stream accuracy Joint : Bone = 1 : 4")
print(accuracy_score_2s_weight4)


### Cross Subject benchmark: Two stream experiments - (Joint, Joint Motion)
action_scores_jjm_2s = []
for i in range(len(joint_out_pd)):
    action_scores_jjm_2s.append(joint_out_pd[i] + jm_out_pd[i])

action_label_jjm_2s = []
for i in range(len(action_scores_jjm_2s)):
    action_label_jjm_2s.append(np.argmax(action_scores_jjm_2s[i]))

print("Two-stream accuracy Joint : Joint Motion = 1 : 1")
accuracy_score_jjm_2s = sum(1 for x,y in zip(action_label_jjm_2s, label) if x == y) / float(len(action_label_jjm_2s))
print(accuracy_score_jjm_2s)


### Cross Subject benchmark: Two stream experiments - (Joint, Bone Motion)
action_scores_jbm_2s = []
for i in range(len(joint_out_pd)):
    action_scores_jbm_2s.append(joint_out_pd[i] + bm_out_pd[i])

action_label_jbm_2s = []
for i in range(len(action_scores_jbm_2s)):
    action_label_jbm_2s.append(np.argmax(action_scores_jbm_2s[i]))

print("Two-stream accuracy Joint : Bone Motion = 1 : 1")
accuracy_score_jbm_2s = sum(1 for x,y in zip(action_label_jbm_2s, label) if x == y) / float(len(action_label_jbm_2s))
print(accuracy_score_jbm_2s)


### Cross Subject benchmark: Two stream experiments - (Bone, Joint Motion)
action_scores_bjm_2s = []
for i in range(len(joint_out_pd)):
    action_scores_bjm_2s.append(bone_out_pd[i] + jm_out_pd[i])

action_label_bjm_2s = []
for i in range(len(action_scores_bjm_2s)):
    action_label_bjm_2s.append(np.argmax(action_scores_bjm_2s[i]))

print("Two-stream accuracy Bone : Joint Motion = 1 : 1")
accuracy_score_bjm_2s = sum(1 for x,y in zip(action_label_bjm_2s, label) if x == y) / float(len(action_label_bjm_2s))
print(accuracy_score_bjm_2s)


### Cross Subject benchmark: Two stream experiments - (Bone, Bone Motion)
action_scores_bbm_2s = []
for i in range(len(joint_out_pd)):
    action_scores_bbm_2s.append(bone_out_pd[i] + bm_out_pd[i])

action_label_bbm_2s = []
for i in range(len(action_scores_bbm_2s)):
    action_label_bbm_2s.append(np.argmax(action_scores_bbm_2s[i]))

print("Two-stream accuracy Bone : Bone Motion = 1 : 1")
accuracy_score_bbm_2s = sum(1 for x,y in zip(action_label_bbm_2s, label) if x == y) / float(len(action_label_bbm_2s))
print(accuracy_score_bbm_2s)


### Cross Subject benchmark: Two stream experiments - (Joint Motion, Bone Motion)
action_scores_jmbm_2s = []
for i in range(len(joint_out_pd)):
    action_scores_jmbm_2s.append(jm_out_pd[i] + bm_out_pd[i])

action_label_jmbm_2s = []
for i in range(len(action_scores_jmbm_2s)):
    action_label_jmbm_2s.append(np.argmax(action_scores_jmbm_2s[i]))

print("Two-stream accuracy Joint Motion : Bone Motion = 1 : 1")
accuracy_score_jmbm_2s = sum(1 for x,y in zip(action_label_jmbm_2s, label) if x == y) / float(len(action_label_jmbm_2s))
print(accuracy_score_jmbm_2s)


### Cross Subject benchmark: Three stream experiments
# Three stream: Joint : Bone : Joint Motion
action_scores_3s_jbjm = []
for i in range(len(joint_out_pd)):
    action_scores_3s_jbjm.append(joint_out_pd[i] + bone_out_pd[i] + jm_out_pd[i])

action_label_3s_jbjm = []
for i in range(len(action_scores_3s_jbjm)):
    action_label_3s_jbjm.append(np.argmax(action_scores_3s_jbjm[i]))

print("Three-stream accuracy Joint : Bone : JM = 1 : 1 : 1")
accuracy_score_3s_jbjm = sum(1 for x,y in zip(action_label_3s_jbjm, label) if x == y) / float(len(action_label_3s_jbjm))
print(accuracy_score_3s_jbjm)

# Three stream: Joint : Bone : Bone Motion
action_scores_3s_jbbm = []
for i in range(len(joint_out_pd)):
    action_scores_3s_jbbm.append(joint_out_pd[i] + bone_out_pd[i] + bm_out_pd[i])

action_label_3s_jbbm = []
for i in range(len(action_scores_3s_jbbm)):
    action_label_3s_jbbm.append(np.argmax(action_scores_3s_jbbm[i]))

print("Three-stream accuracy Joint : Bone : BM = 1 : 1 : 1")
accuracy_score_3s_jbbm = sum(1 for x,y in zip(action_label_3s_jbbm, label) if x == y) / float(len(action_label_3s_jbbm))
print(accuracy_score_3s_jbbm)


### Cross Subject benchmark: Four stream weights experiments
# Four stream: Joint : Bone : JM : BM = 1 : 1 : 1 : 1
action_scores_4s = []
for i in range(len(joint_out_pd)):
    action_scores_4s.append(joint_out_pd[i] + bone_out_pd[i] + jm_out_pd[i] + bm_out_pd[i])

action_label_4s = []
for i in range(len(action_scores_4s)):
    action_label_4s.append(np.argmax(action_scores_4s[i]))

print("Four-stream accuracy Joint : Bone : JM : BM = 1 : 1 : 1 : 1")
accuracy_score_4s = sum(1 for x,y in zip(action_label_4s, label) if x == y) / float(len(action_label_4s))
print(accuracy_score_4s)

# Four stream: Joint : Bone : JM : BM = 1 : 2 : 1 : 1
action_scores_4s_12 = []
for i in range(len(joint_out_pd)):
    action_scores_4s_12.append(1 * joint_out_pd[i] + 2 * bone_out_pd[i] + jm_out_pd[i] + bm_out_pd[i])

action_label_4s_12 = []
for i in range(len(action_scores_4s_12)):
    action_label_4s_12.append(np.argmax(action_scores_4s_12[i]))

print("Four-stream accuracy Joint : Bone : JM : BM = 1 : 2 : 1 : 1")
accuracy_score_4s_12 = sum(1 for x,y in zip(action_label_4s_12, label) if x == y) / float(len(action_label_4s_12))
print(accuracy_score_4s_12)


# Four stream: Joint : Bone : JM : BM = 1 : 3 : 1 : 1
action_scores_4s_13 = []
for i in range(len(joint_out_pd)):
    action_scores_4s_13.append(1 * joint_out_pd[i] + 3 * bone_out_pd[i] + jm_out_pd[i] + bm_out_pd[i])

action_label_4s_13 = []
for i in range(len(action_scores_4s_13)):
    action_label_4s_13.append(np.argmax(action_scores_4s_13[i]))

print("Four-stream accuracy Joint : Bone : JM : BM = 1 : 3 : 1 : 1")
accuracy_score_4s_13 = sum(1 for x,y in zip(action_label_4s_13, label) if x == y) / float(len(action_label_4s_13))
print(accuracy_score_4s_13)

# Four stream: Joint : Bone : JM : BM = 2 : 2 : 1 : 1
action_scores_4s_2 = []
for i in range(len(joint_out_pd)):
    action_scores_4s_2.append(2 * joint_out_pd[i] + 2 * bone_out_pd[i] + jm_out_pd[i] + bm_out_pd[i])

action_label_4s_2 = []
for i in range(len(action_scores_4s_2)):
    action_label_4s_2.append(np.argmax(action_scores_4s_2[i]))

print("Four-stream accuracy Joint : Bone : JM : BM = 2 : 2 : 1 : 1")
accuracy_score_4s_2 = sum(1 for x,y in zip(action_label_4s_2, label) if x == y) / float(len(action_label_4s_2))
print(accuracy_score_4s_2)

# Four stream: Joint : Bone : JM : BM = 2 : 3 : 1 : 1
action_scores_4s_23 = []
for i in range(len(joint_out_pd)):
    action_scores_4s_23.append(2 * joint_out_pd[i] + 3 * bone_out_pd[i] + jm_out_pd[i] + bm_out_pd[i])

action_label_4s_23 = []
for i in range(len(action_scores_4s_23)):
    action_label_4s_23.append(np.argmax(action_scores_4s_23[i]))

print("Four-stream accuracy Joint : Bone : JM : BM = 2 : 3 : 1 : 1")
accuracy_score_4s_23 = sum(1 for x,y in zip(action_label_4s_23, label) if x == y) / float(len(action_label_4s_23))
print(accuracy_score_4s_23) 

# Four stream: Joint : Bone : JM : BM = 2 : 4 : 1 : 1
action_scores_4s_24 = []
for i in range(len(joint_out_pd)):
    action_scores_4s_24.append(2 * joint_out_pd[i] + 4 * bone_out_pd[i] + jm_out_pd[i] + bm_out_pd[i])

action_label_4s_24 = []
for i in range(len(action_scores_4s_24)):
    action_label_4s_24.append(np.argmax(action_scores_4s_24[i]))

print("Four-stream accuracy Joint : Bone : JM : BM = 2 : 4 : 1 : 1")
accuracy_score_4s_24 = sum(1 for x,y in zip(action_label_4s_24, label) if x == y) / float(len(action_label_4s_24))
print(accuracy_score_4s_24)

