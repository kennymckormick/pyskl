# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import socket

from mmcv.utils import get_logger


def mc_on(port=22077, launcher='pytorch', size=24000):
    # size is mb, allocate 24GB memory by default.
    mc_exe = 'memcached' if launcher == 'pytorch' else '/mnt/lustre/share/memcached/bin/memcached'
    os.system(f'{mc_exe} -p {port} -m {size}m -d')


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
