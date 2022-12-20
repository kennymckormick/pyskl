import copy as cp
import multiprocessing as mp
import numpy as np
import os
import os.path as osp
from mmcv import dump
from tqdm import tqdm

from pyskl.smp import mrlines

eps = 1e-3


def parse_skeleton_file(ske_name, root='nturgb+d_skeletons'):
    ske_file = osp.join(root, ske_name + '.skeleton')

    lines = mrlines(ske_file)
    idx = 0
    num_frames = int(lines[0])
    num_joints = 25
    idx += 1

    body_data = dict()
    fidx = 0

    for f in range(num_frames):
        num_bodies = int(lines[idx])
        idx += 1
        if num_bodies == 0:
            continue
        for b in range(num_bodies):
            bodyID = int(lines[idx].split()[0])
            if bodyID not in body_data:
                kpt = []
                body_data[bodyID] = dict(kpt=kpt, start=fidx)
            idx += 1
            assert int(lines[idx]) == 25
            idx += 1
            joints = np.zeros((25, 3), dtype=np.float32)

            for j in range(num_joints):
                line = lines[idx].split()
                joints[j, :3] = np.array(line[:3], dtype=np.float32)
                idx += 1
            body_data[bodyID]['kpt'].append(joints)
        fidx += 1

    for k in body_data:
        body_data[k]['motion'] = np.sum(np.var(np.vstack(body_data[k]['kpt']), axis=0))
        body_data[k]['kpt'] = np.stack(body_data[k]['kpt'])

    assert idx == len(lines)
    return body_data


def spread_denoising(body_data_list):
    wh_ratio = 0.8
    spnoise_ratio = 0.69754

    def get_valid_frames(kpt):
        valid_frames = []
        for i in range(kpt.shape[0]):
            x, y = kpt[i, :, 0], kpt[i, :, 1]
            if (x.max() - x.min()) <= wh_ratio * (y.max() - y.min()):
                valid_frames.append(i)
        return valid_frames

    for item in body_data_list:
        valid_frames = get_valid_frames(item['kpt'])
        if len(valid_frames) == item['kpt'].shape[0]:
            item['flag'] = True
            continue
        ratio = len(valid_frames) / item['kpt'].shape[0]
        if 1 - ratio >= spnoise_ratio:
            item['flag'] = False
        else:
            item['flag'] = True
            item['motion'] = min(item['motion'],
                                 np.sum(np.var(item['kpt'][valid_frames].reshape(-1, 3), axis=0)))
    body_data_list = [item for item in body_data_list if item['flag']]
    assert len(body_data_list) >= 1
    _ = [item.pop('flag') for item in body_data_list]
    body_data_list.sort(key=lambda x: -x['motion'])
    return body_data_list


def non_zero(kpt):
    s = 0
    e = kpt.shape[1]
    while np.sum(np.abs(kpt[:, s])) < eps:
        s += 1
    while np.sum(np.abs(kpt[:, e - 1])) < eps:
        e -= 1
    return kpt[:, s: e]


def gen_keypoint_array(body_data):
    length_threshold = 11

    body_data = cp.deepcopy(list(body_data.values()))
    body_data.sort(key=lambda x: -x['motion'])
    if len(body_data) == 1:
        return body_data[0]['kpt'][None]
    else:
        body_data = [item for item in body_data if item['kpt'].shape[0] > length_threshold]
        if len(body_data) == 1:
            return body_data[0]['kpt'][None]
        body_data = spread_denoising(body_data)
        if len(body_data) == 1:
            return body_data[0]['kpt'][None]
        max_fidx = 0

        for item in body_data:
            max_fidx = max(max_fidx, item['start'] + item['kpt'].shape[0])
        keypoint = np.zeros((2, max_fidx, 25, 3), np.float32)

        s1, e1, s2, e2 = body_data[0]['start'], body_data[0]['start'] + body_data[0]['kpt'].shape[0], 0, 0
        keypoint[0, s1: e1] = body_data[0]['kpt']
        for item in body_data[1:]:
            s, e = item['start'], item['start'] + item['kpt'].shape[0]
            if max(s1, s) >= min(e1, e):
                keypoint[0, s: e] = item['kpt']
                s1, e1 = min(s, s1), max(e, e1)
            elif max(s2, s) >= min(e2, e):
                keypoint[1, s: e] = item['kpt']
                s2, e2 = min(s, s2), max(e, e2)

        keypoint = non_zero(keypoint)
        if np.sum(np.abs(keypoint[0, 0, 1])) < eps and np.sum(np.abs(keypoint[1, 0, 1])) > eps:
            keypoint = keypoint[::-1]
        return keypoint


root = 'nturgb+d_skeletons'
skeleton_files = os.listdir(root)
names = [x.split('.')[0] for x in skeleton_files]
names.sort()
missing = mrlines('ntu120_missing.txt')
missing = set(missing)
names = [x for x in names if x not in missing]

extended = False
for name in names:
    if int(name.split('A')[-1]) > 60:
        extended = True
        print('NTURGB+D 120 skeleton files detected, will generate both ntu60_3danno.pkl and ntu120_3danno.pkl. ')
        break

if not extended:
    print('NTURGB+D 120 skeleton files not detected, will only generate ntu60_3danno.pkl. ')


def gen_anno(name):
    body_data = parse_skeleton_file(name, root)
    if len(body_data) == 0:
        return None
    keypoint = gen_keypoint_array(body_data).astype(np.float16)
    label = int(name.split('A')[-1]) - 1
    total_frames = keypoint.shape[1]
    return dict(frame_dir=name, label=label, keypoint=keypoint, total_frames=total_frames)


anno_dict = {}
num_process = 1

if num_process == 1:
    # Each annotations has 4 keys: frame_dir, label, keypoint, total_frames
    for name in tqdm(names):
        anno_dict[name] = gen_anno(name)
else:
    pool = mp.Pool(num_process)
    annotations = pool.map(gen_anno, names)
    pool.close()
    for anno in annotations:
        anno_dict[anno['frame_dir']] = anno

names = [x for x in names if anno_dict is not None]
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
    38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
    80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
]

if extended:
    xsub_train = [name for name in names if int(name.split('P')[1][:3]) in training_subjects]
    xsub_val = [name for name in names if int(name.split('P')[1][:3]) not in training_subjects]
    xset_train = [name for name in names if int(name.split('S')[1][:3]) % 2 == 0]
    xset_val = [name for name in names if int(name.split('S')[1][:3]) % 2 == 1]
    split = dict(xsub_train=xsub_train, xsub_val=xsub_val, xset_train=xset_train, xset_val=xset_val)
    annotations = [anno_dict[name] for name in names]
    dump(dict(split=split, annotations=annotations), 'ntu120_3danno.pkl')

names = [name for name in names if int(name.split('A')[-1]) <= 60]
xsub_train = [name for name in names if int(name.split('P')[1][:3]) in training_subjects]
xsub_val = [name for name in names if int(name.split('P')[1][:3]) not in training_subjects]
xview_train = [name for name in names if 'C001' not in name]
xview_val = [name for name in names if 'C001' in name]
split = dict(xsub_train=xsub_train, xsub_val=xsub_val, xview_train=xview_train, xview_val=xview_val)
annotations = [anno_dict[name] for name in names]
dump(dict(split=split, annotations=annotations), 'ntu60_3danno.pkl')
