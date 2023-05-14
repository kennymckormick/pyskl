from __future__ import print_function, absolute_import

import pandas as pd
import numpy as np
import pickle


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


list_xsub_val_names = data[0]['split']['xview_val']
print(len(list_xsub_val_names))
print(list_xsub_val_names[0])
label = []

print(len(data[0]['annotations']))

for annotation in range(len(data[0]['annotations'])):
    if data[0]['annotations'][annotation]['frame_dir'] in list_xsub_val_names:
        label.append(data[0]['annotations'][annotation]['label'])

print(len(label))
# print(label)

### Cross view benchmark
with (open("stgcn/stgcn_pyskl_ntu60_xview_hrnet/j_result.pkl", "rb")) as openfile:
    joint_out_cv = pickle.load(openfile)


with (open("stgcn/stgcn_pyskl_ntu60_xview_hrnet/b_result.pkl", "rb")) as openfile:
    bone_out_cv = pickle.load(openfile)

with (open("stgcn/stgcn_pyskl_ntu60_xview_hrnet/jm_result.pkl", "rb")) as openfile:
    jm_out_cv = pickle.load(openfile)

with (open("stgcn/stgcn_pyskl_ntu60_xview_hrnet/bm_result.pkl", "rb")) as openfile:
    bm_out_cv = pickle.load(openfile)


### Cross view benchmark two stream experiments
# Two stream: Joint : Bone = 1 : 1
action_scores_2s = []
for i in range(len(joint_out_cv)):
    action_scores_2s.append(joint_out_cv[i] + bone_out_cv[i])

action_label_2s = []
for i in range(len(action_scores_2s)):
    action_label_2s.append(np.argmax(action_scores_2s[i]))

accuracy_score_2s = sum(1 for x, y in zip(action_label_2s, label) if x == y) / float(len(action_label_2s))
print("Two stream accuraycy Joint : Bone = 1 : 1")
print(accuracy_score_2s) 

# Two stream: Joint : Bone = 2 : 1
action_scores_2s_2 = []
for i in range(len(joint_out_cv)):
    action_scores_2s_2.append(2 * joint_out_cv[i] + bone_out_cv[i])

action_label_2s_2 = []
for i in range(len(action_scores_2s_2)):
    action_label_2s_2.append(np.argmax(action_scores_2s_2[i]))

accuracy_score_2s_2 = sum(1 for x, y in zip(action_label_2s_2, label) if x == y) / float(len(action_label_2s_2))
print("Two stream accuraycy Joint : Bone = 2 : 1")
print(accuracy_score_2s_2) 

# Two stream: Joint : Bone = 3 : 1
action_scores_2s_3 = []
for i in range(len(joint_out_cv)):
    action_scores_2s_3.append(3 * joint_out_cv[i] + bone_out_cv[i])

action_label_2s_3 = []
for i in range(len(action_scores_2s_3)):
    action_label_2s_3.append(np.argmax(action_scores_2s_3[i]))

accuracy_score_2s_3 = sum(1 for x, y in zip(action_label_2s_3, label) if x == y) / float(len(action_label_2s_3))
print("Two stream accuraycy Joint : Bone = 3 : 1")
print(accuracy_score_2s_3) 

# Two stream: Joint : Bone = 4 : 1
action_scores_2s_4 = []
for i in range(len(joint_out_cv)):
    action_scores_2s_4.append(4 * joint_out_cv[i] + bone_out_cv[i])

action_label_2s_4 = []
for i in range(len(action_scores_2s_4)):
    action_label_2s_4.append(np.argmax(action_scores_2s_4[i]))

accuracy_score_2s_4 = sum(1 for x, y in zip(action_label_2s_4, label) if x == y) / float(len(action_label_2s_4))
print("Two stream accuraycy Joint : Bone = 4 : 1")
print(accuracy_score_2s_4) 

# Two stream: Joint : Joint Motion
action_scores_2s_jjm = []
for i in range(len(joint_out_cv)):
    action_scores_2s_jjm.append(joint_out_cv[i] + jm_out_cv[i])

action_label_2s_jjm = []
for i in range(len(action_scores_2s_jjm)):
    action_label_2s_jjm.append(np.argmax(action_scores_2s_jjm[i]))

accuracy_score_2s_jjm = sum(1 for x, y in zip(action_label_2s_jjm, label) if x == y) / float(len(action_label_2s_jjm))
print("Two stream accuraycy Joint : Joint Motion")
print(accuracy_score_2s_jjm) 

# Two stream: Joint : Bone Motion
action_scores_2s_jbm = []
for i in range(len(joint_out_cv)):
    action_scores_2s_jbm.append(joint_out_cv[i] + bm_out_cv[i])

action_label_2s_jbm = []
for i in range(len(action_scores_2s_jbm)):
    action_label_2s_jbm.append(np.argmax(action_scores_2s_jbm[i]))

accuracy_score_2s_jbm = sum(1 for x, y in zip(action_label_2s_jbm, label) if x == y) / float(len(action_label_2s_jbm))
print("Two stream accuraycy Joint : Bone Motion")
print(accuracy_score_2s_jbm)

# Two stream: Bone : Joint Motion
action_scores_2s_bjm = []
for i in range(len(joint_out_cv)):
    action_scores_2s_bjm.append(bone_out_cv[i] + jm_out_cv[i])

action_label_2s_bjm = []
for i in range(len(action_scores_2s_bjm)):
    action_label_2s_bjm.append(np.argmax(action_scores_2s_bjm[i]))

accuracy_score_2s_bjm = sum(1 for x, y in zip(action_label_2s_bjm, label) if x == y) / float(len(action_label_2s_bjm))
print("Two stream accuraycy Bone : Joint Motion")
print(accuracy_score_2s_bjm)

# Two stream: Bone : Bone Motion
action_scores_2s_bbm = []
for i in range(len(joint_out_cv)):
    action_scores_2s_bbm.append(bone_out_cv[i] + bm_out_cv[i])

action_label_2s_bbm = []
for i in range(len(action_scores_2s_bbm)):
    action_label_2s_bbm.append(np.argmax(action_scores_2s_bbm[i]))

accuracy_score_2s_bbm = sum(1 for x, y in zip(action_label_2s_bbm, label) if x == y) / float(len(action_label_2s_bbm))
print("Two stream accuraycy Bone : Bone Motion")
print(accuracy_score_2s_bbm)

# Two stream: Joint Motion : Bone Motion
action_scores_2s_jmbm = []
for i in range(len(joint_out_cv)):
    action_scores_2s_jmbm.append(jm_out_cv[i] + bm_out_cv[i])

action_label_2s_jmbm = []
for i in range(len(action_scores_2s_jmbm)):
    action_label_2s_jmbm.append(np.argmax(action_scores_2s_jmbm[i]))

accuracy_score_2s_jmbm = sum(1 for x, y in zip(action_label_2s_jmbm, label) if x == y) / float(len(action_label_2s_jmbm))
print("Two stream accuraycy Joint Motion : Bone Motion")
print(accuracy_score_2s_jmbm)

### Cross view three stream experiments
# Three stream: Joint : Bone : Joint Motion
action_scores_2s_jbjm = []
for i in range(len(joint_out_cv)):
    action_scores_2s_jbjm.append(joint_out_cv[i] + bone_out_cv[i] + jm_out_cv[i])

action_label_2s_jbjm = []
for i in range(len(action_scores_2s_jbjm)):
    action_label_2s_jbjm.append(np.argmax(action_scores_2s_jbjm[i]))

accuracy_score_2s_jbjm = sum(1 for x, y in zip(action_label_2s_jbjm, label) if x == y) / float(len(action_label_2s_jbjm))
print("Three stream accuraycy Joint : Bone : Joint Motion")
print(accuracy_score_2s_jbjm)

# Three stream: Joint : Bone : Bone Motion
action_scores_2s_jbbm = []
for i in range(len(joint_out_cv)):
    action_scores_2s_jbbm.append(joint_out_cv[i] + bone_out_cv[i] + bm_out_cv[i])

action_label_2s_jbbm = []
for i in range(len(action_scores_2s_jbbm)):
    action_label_2s_jbbm.append(np.argmax(action_scores_2s_jbbm[i]))

accuracy_score_2s_jbbm = sum(1 for x, y in zip(action_label_2s_jbbm, label) if x == y) / float(len(action_label_2s_jbbm))
print("Three stream accuraycy Joint : Bone : Bone Motion")
print(accuracy_score_2s_jbbm)


### Cross View Benchmark Four stream experiments
# Four stream: Joint : Bone : JM : BM = 1 : 1 : 1 : 1
action_scores_4s = []
for i in range(len(joint_out_cv)):
    action_scores_4s.append(joint_out_cv[i] + bone_out_cv[i] + jm_out_cv[i] + bm_out_cv[i])

action_label_4s = []
for i in range(len(action_scores_4s)):
    action_label_4s.append(np.argmax(action_scores_4s[i]))

accuracy_score_4s = sum(1 for x, y in zip(action_label_4s, label) if x == y) / float(len(action_label_4s))
print("Four stream accuraycy Joint : Bone : JM : BM = 1 : 1 : 1 : 1")
print(accuracy_score_4s)

# Four stream: Joint : Bone : JM : BM = 2 : 1 : 1 : 1
action_scores_4s_21 = []
for i in range(len(joint_out_cv)):
    action_scores_4s_21.append(2 * joint_out_cv[i] + bone_out_cv[i] + jm_out_cv[i] + bm_out_cv[i])

action_label_4s_21 = []
for i in range(len(action_scores_4s_21)):
    action_label_4s_21.append(np.argmax(action_scores_4s_21[i]))

accuracy_score_4s_21 = sum(1 for x, y in zip(action_label_4s_21, label) if x == y) / float(len(action_label_4s_21))
print("Four stream accuraycy Joint : Bone : JM : BM = 2 : 1 : 1 : 1")
print(accuracy_score_4s_21)

# Four stream: Joint : Bone : JM : BM = 3 : 1 : 1 : 1
action_scores_4s_31 = []
for i in range(len(joint_out_cv)):
    action_scores_4s_31.append(3 * joint_out_cv[i] + bone_out_cv[i] + jm_out_cv[i] + bm_out_cv[i])

action_label_4s_31 = []
for i in range(len(action_scores_4s_21)):
    action_label_4s_31.append(np.argmax(action_scores_4s_31[i]))

accuracy_score_4s_31 = sum(1 for x, y in zip(action_label_4s_31, label) if x == y) / float(len(action_label_4s_31))
print("Four stream accuraycy Joint : Bone : JM : BM = 3 : 1 : 1 : 1")
print(accuracy_score_4s_31)

# Four stream: Joint : Bone : JM : BM = 4 : 1 : 1 : 1
# action_scores_4s_41 = []
# for i in range(len(joint_out_cv)):
#     action_scores_4s_41.append(4 * joint_out_cv[i] + bone_out_cv[i] + jm_out_cv[i] + bm_out_cv[i])

# action_label_4s_41 = []
# for i in range(len(action_scores_4s_41)):
#     action_label_4s_41.append(np.argmax(action_scores_4s_41[i]))

# accuracy_score_4s_41 = sum(1 for x, y in zip(action_label_4s_41, label) if x == y) / float(len(action_label_4s_41))
# print("Four stream accuraycy Joint : Bone : JM : BM = 4 : 1 : 1 : 1")
# print(accuracy_score_4s_41)

# Four stream: Joint : Bone : JM : BM = 2 : 2 : 1 : 1
action_scores_4s_22 = []
for i in range(len(joint_out_cv)):
    action_scores_4s_22.append(2 * joint_out_cv[i] + 2 * bone_out_cv[i] + jm_out_cv[i] + bm_out_cv[i])

action_label_4s_22 = []
for i in range(len(action_scores_4s_22)):
    action_label_4s_22.append(np.argmax(action_scores_4s_22[i]))

accuracy_score_4s_22 = sum(1 for x, y in zip(action_label_4s_22, label) if x == y) / float(len(action_label_4s_22))
print("Four stream accuraycy Joint : Bone : JM : BM = 2 : 2 : 1 : 1")
print(accuracy_score_4s_22)

# Four stream: Joint : Bone : JM : BM = 3 : 2 : 1 : 1
action_scores_4s_32 = []
for i in range(len(joint_out_cv)):
    action_scores_4s_32.append(3 * joint_out_cv[i] + 2 * bone_out_cv[i] + jm_out_cv[i] + bm_out_cv[i])

action_label_4s_32 = []
for i in range(len(action_scores_4s_32)):
    action_label_4s_32.append(np.argmax(action_scores_4s_32[i]))

accuracy_score_4s_32 = sum(1 for x, y in zip(action_label_4s_32, label) if x == y) / float(len(action_label_4s_32))
print("Four stream accuraycy Joint : Bone : JM : BM = 3 : 2 : 1 : 1")
print(accuracy_score_4s_32)

# Four stream: Joint : Bone : JM : BM = 4 : 2 : 1 : 1
action_scores_4s_42 = []
for i in range(len(joint_out_cv)):
    action_scores_4s_42.append(4 * joint_out_cv[i] + 2 * bone_out_cv[i] + jm_out_cv[i] + bm_out_cv[i])

action_label_4s_42 = []
for i in range(len(action_scores_4s_42)):
    action_label_4s_42.append(np.argmax(action_scores_4s_42[i]))

accuracy_score_4s_42 = sum(1 for x, y in zip(action_label_4s_42, label) if x == y) / float(len(action_label_4s_42))
print("Four stream accuraycy Joint : Bone : JM : BM = 4 : 2 : 1 : 1")
print(accuracy_score_4s_42)
