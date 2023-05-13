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


# Cross view benchmark two stream experiments
action_scores_2s_cv = []
for i in range(len(joint_out_cv)):
    action_scores_2s_cv.append(joint_out_cv[i] + bone_out_cv[i])


action_label_2s_cv = []
for i in range(len(action_scores_2s_cv)):
    action_label_2s_cv.append(np.argmax(action_scores_2s_cv[i]))

print(len(action_label_2s_cv))

accuracy_score_2s_cv = sum(1 for x, y in zip(action_label_2s_cv, label) if x == y) / float(len(action_label_2s_cv))
print("Cross View benchmark Two stream accuraycy Joint : Bone = 1 : 1")
print(accuracy_score_2s_cv) 


action_scores_2s_cv_2 = []
for i in range(len(joint_out_cv)):
    action_scores_2s_cv_2.append(2 * joint_out_cv[i] + bone_out_cv[i])

action_label_2s_cv_2 = []
for i in range(len(action_scores_2s_cv_2)):
    action_label_2s_cv_2.append(np.argmax(action_scores_2s_cv_2[i]))

accuracy_score_2s_cv_2 = sum(1 for x, y in zip(action_label_2s_cv_2, label) if x == y) / float(len(action_label_2s_cv_2))
print("Cross View benchmark Two stream accuraycy Joint : Bone = 2 : 1")
print(accuracy_score_2s_cv_2) 


