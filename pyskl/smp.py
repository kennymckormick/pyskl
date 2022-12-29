# flake8: noqa: F401, F403
import abc
import argparse
import collections
import cv2
import json
import multiprocessing as mp
import numpy as np
import os
import os.path as osp
import pickle
import random as rd
import requests
import shutil
import string
import subprocess
import sys
import time
import warnings
from collections import OrderedDict, defaultdict
from functools import reduce
from fvcore.nn import FlopCountAnalysis, parameter_count
from multiprocessing import Pool, current_process
from tqdm import tqdm

try:
    import decord
except ImportError:
    pass


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

def youtube_dl(idx):
    cmd = f'youtube-dl -f best -f mp4 "{idx}"  -o {idx}.mp4'
    os.system(cmd)

def run_command(cmd):
    return subprocess.check_output(cmd)

def ls(dirname='.', full=True, match=''):
    if not full or dirname == '.':
        ans = os.listdir(dirname)
    ans = [osp.join(dirname, x) for x in os.listdir(dirname)]
    ans = [x for x in ans if match in x]
    return ans

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

def auto_mix2(scores):
    assert len(scores) == 2
    return {'1:1': comb(scores, [1, 1]), '2:1': comb(scores, [2, 1]), '1:2': comb(scores, [1, 2])}

def top1(score, label):
    return np.mean(intop(score, label, 1))

def topk(score, label, k=1):
    return np.mean(intop(score, label, k)) if isinstance(k, int) else [topk(score, label, kk) for kk in k]

def load_label(ann, split=None):
    if ann.endswith('.txt'):
        lines = mrlines(ann)
        return [int(x.split()[-1]) for x in lines]
    elif ann.endswith('.pkl'):
        data = lpkl(ann)
        if split is not None:
            split = set(data['split'][split])
            assert 'annos' in data or 'annotations' in data
            annotations = data['annos'] if 'annos' in data else data['annotations']
            key_name = 'frame_dir' if 'frame_dir' in annotations[0] else 'filename'
            data = [x for x in annotations if x[key_name] in split]
        return [x['label'] for x in data]
    else:
        raise NotImplemented

def mean_acc(pred, label, with_class_acc=False):
    hits = defaultdict(list)
    for p, g in zip(pred, label):
        hits[g].append(np.argmax(p) == g)
    keys = list(hits.keys())
    keys.sort()
    class_acc = [np.mean(hits[k]) for k in keys]
    return (np.mean(class_acc), class_acc) if with_class_acc else np.mean(class_acc)

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
    cmds_main = []
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
        cmds_main.append('  &&  '.join(cmds) + '  &')
    timestamp = time.strftime('%m%d%H%M%S', time.localtime())
    mwlines(cmds_main, f'train_{timestamp}.sh')

def h2r(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def r2h(rgb):
    return '#%02x%02x%02x' % rgb

def fnp(model, input=None):
    params = parameter_count(model)['']
    print('Parameter Size: {:.4f} M'.format(params / 1024 / 1024))
    if input is not None:
        flops = FlopCountAnalysis(model, input).total()
        print('FLOPs: {:.4f} G'.format(flops / 1024 / 1024 / 1024))
        return params, flops
    return params, None

def cache_objects(mc_root, mc_cfg=('localhost', 22077), mc_size=60000, num_proc=32):
    from pyskl.utils import mc_on, mp_cache, mp_cache_single, test_port
    assert isinstance(mc_cfg, tuple) and mc_cfg[0] == 'localhost'
    if not test_port(mc_cfg[0], mc_cfg[1]):
        mc_on(port=mc_cfg[1], launcher='pytorch', size=mc_size)
    retry = 3
    while not test_port(mc_cfg[0], mc_cfg[1]) and retry > 0:
        time.sleep(5)
        retry -= 1
    assert retry >= 0, 'Failed to launch memcached. '
    # Add a pre-fetch step
    if osp.isdir(mc_root):
        files = ls(mc_root)
        mp_cache(mc_cfg, files, num_proc=num_proc)
    elif osp.isfile(mc_root):
        mp_cache_single(mc_cfg, mc_root, num_proc=num_proc)
