# flake8: noqa: F401, F403
import abc
import argparse
import collections
import json
import multiprocessing as mp
import os
import os.path as osp
import pickle
import random as rd
import shutil
import string
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict
from functools import reduce
from multiprocessing import Pool, current_process

import cv2
import decord
import mmcv
import numpy as np
import requests
from tqdm import tqdm


def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f

def mwlines(lines, fname):
    with open(fname, 'w') as fout:
        fout.write('\n'.join(lines))

def default_set(self, args, name, default):
    if hasattr(args, name):
        val = getattr(args, name)
        setattr(self, name, val)
    else:
        setattr(self, name, default)

def youtube_dl(url, output_name):
    cmd = 'youtube-dl -f best -f mp4 "{}"  -o {}'.format(url, output_name)
    os.system(cmd)

def run_command(cmd):
    return subprocess.check_output(cmd)

def ls(dirname='.', full=True):
    if not full:
        return os.listdir(dirname)
    return [osp.join(dirname, x) for x in os.listdir(dirname)]

def add(x, y):
    return x + y

def lpkl(pth):
    return pickle.load(open(pth, 'rb'))

def ljson(pth):
    return json.load(open(pth, 'r'))

def intop(pred, label, n):
    pred = [np.argsort(x)[-n:] for x in pred]
    hit = [(l in p) for l, p in zip(label, pred)]
    return hit

def comb(scores, coeffs):
    ret = [x * coeffs[0] for x in scores[0]]
    for i in range(1, len(scores)):
        ret = [x + y for x, y in zip(ret, [x * coeffs[i] for x in scores[i]])]
    return ret

def top1(score, label):
    return np.mean(intop(score, label, 1))

def load_label(ann, split=None):
    if ann.endswith('.txt'):
        lines = mrlines(ann)
        return [int(x.split()[-1]) for x in lines]
    elif ann.endswith('.pkl'):
        data = lpkl(ann)
        if split is not None:
            split, annotations = set(data['split'][split]), data['annotations']
            key_name = 'frame_dir' if 'frame_dir' in annotations[0] else 'filename'
            data = [x for x in annotations if x[key_name] in split]
        return [x['label'] for x in data]
    else:
        raise NotImplemented

def mean_acc(pred, label, with_class_acc=False):
    hits = defaultdict(list)
    for p, g in zip(pred, label):
        hits[g].append(np.argmax(p) == g)
    class_acc = [np.mean(x) for x in hits.values()]
    return np.mean(class_acc), class_acc if with_class_acc else np.mean(class_acc)

def match_dict(s, d):
    values = []
    for k, v in d.items():
        if k in s:
            values.append(v)
    assert len(values) == 1
    return values[0]

def download_file(url, filename=None):
    if filename is None:
        filename = url.split('/')[-1]
    response = requests.get(url)
    open(filename, 'wb').write(response.content)

def gen_bash(cfgs, num_gpus, gpus_per_task=1):
    rd.shuffle(cfgs)
    num_bash = num_gpus // gpus_per_task
    for i in range(num_bash):
        cmds = []
        for c in cfgs[i::num_bash]:
            port = rd.randint(30000, 50000)
            gpu_ids = list(range(i, num_gpus, num_bash))
            gpu_ids = ','.join([str(x) for x in gpu_ids])
            cmds.append(
                f'CUDA_VISIBLE_DEVICES={gpu_ids} PORT={port} bash tools/dist_train.sh {c} {gpus_per_task} '
                '--validate --test-last --test-best'
            )
        timestamp = time.strftime('%m%d%H%M%S', time.localtime())
        mwlines(cmds, f'train_{timestamp}_{i}.sh')

def h2r(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def r2h(rgb):
    return '#%02x%02x%02x' % rgb
