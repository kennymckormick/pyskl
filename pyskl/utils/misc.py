# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import hashlib
import logging
import multiprocessing as mp
import numpy as np
import os
import os.path as osp
import socket
import warnings
from mmcv import load
from mmcv.runner import get_dist_info
from mmcv.utils import get_logger

from ..smp import download_file


def mc_on(port=22077, launcher='pytorch', size=60000, min_size=6):
    # size is mb, allocate 24GB memory by default.
    mc_exe = 'memcached' if launcher == 'pytorch' else '/mnt/lustre/share/memcached/bin/memcached'
    os.system(f'{mc_exe} -p {port} -m {size}m -I {min_size}m -d')


def cache_file(arg_tuple):
    mc_cfg, data_file = arg_tuple
    assert isinstance(mc_cfg, tuple) and mc_cfg[0] == 'localhost'
    retry = 3
    while not test_port(mc_cfg[0], mc_cfg[1]) and retry > 0:
        time.sleep(5)
        retry -= 1
    assert retry >= 0, 'Failed to launch memcached. '
    from pymemcache import serde
    from pymemcache.client.base import Client

    cli = Client(mc_cfg, serde=serde.pickle_serde)

    if isinstance(data_file, str):
        assert osp.exists(data_file)
        kv_dict = load(data_file)
    else:
        if not isinstance(data_file, dict):
            assert isinstance(data_file[0], tuple) and len(data_file[0]) == 2
            data_file = {k: v for k, v in data_file}
        kv_dict = data_file

    if isinstance(kv_dict, list):
        assert ('frame_dir' in kv_dict[0]) != ('filename' in kv_dict[0])
        key = 'frame_dir' if 'frame_dir' in kv_dict[0] else 'filename'
        kv_dict = {x[key]: x for x in kv_dict}
    for k, v in kv_dict.items():
        flag = None
        while not isinstance(flag, dict):
            try:
                cli.set(k, v)
            except:
                cli = Client(mc_cfg, serde=serde.pickle_serde)
                cli.set(k, v)
            try:
                flag = cli.get(k)
            except:
                cli = Client(mc_cfg, serde=serde.pickle_serde)
                flag = cli.get(k)


def mp_cache(mc_cfg, mc_list, num_proc=32):
    args = [(mc_cfg, x) for x in mc_list]
    pool = mp.Pool(num_proc)
    pool.map(cache_file, args)


def mp_cache_single(mc_cfg, file_name, num_proc=32):
    data = load(file_name)
    assert 'annotations' in data
    annos = data['annotations']
    tups = [(x['frame_dir'], x) for x in annos]
    tups = [tups[i::num_proc] for i in range(num_proc)]
    args = [(mc_cfg, tup_list) for tup_list in tups]
    pool = mp.Pool(num_proc)
    pool.map(cache_file, args)


def mc_off():
    os.system('killall memcached')


def test_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    assert isinstance(ip, str)
    if isinstance(port, str):
        port = int(port)
    assert 1 <= port <= 65535
    result = sock.connect_ex((ip, port))
    return result == 0


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use ``get_logger`` method in mmcv to get the root logger.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If ``log_file`` is specified, a FileHandler
    will also be added. The name of the root logger is the top-level package
    name, e.g., "pyskl".
    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        :obj:`logging.Logger`: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)


def cache_checkpoint(filename, cache_dir='.cache'):
    if filename.startswith('http://') or filename.startswith('https://'):
        url = filename.split('//')[1]
        basename = filename.split('/')[-1]
        filehash = hashlib.md5(url.encode('utf8')).hexdigest()[-8:]
        os.makedirs(cache_dir, exist_ok=True)
        local_pth = osp.join(cache_dir, basename.replace('.pth', f'_{filehash}.pth'))
        if not osp.exists(local_pth):
            download_file(filename, local_pth)
        filename = local_pth
    return filename

def warning_r0(warn_str):
    rank, _ = get_dist_info()
    if rank == 0:
        warnings.warn(warn_str)
