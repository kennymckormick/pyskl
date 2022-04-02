# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, load
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from pyskl import __version__
from pyskl.apis import init_random_seed, train_model
from pyskl.datasets import build_dataset
from pyskl.models import build_model
from pyskl.utils import collect_env, get_root_logger, mc_off, mc_on, test_port


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test-last',
        action='store_true',
        help='whether to test the checkpoint after training')
    parser.add_argument(
        '--test-best',
        action='store_true',
        help=('whether to test the best checkpoint (if applicable) after '
              'training'))
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if not hasattr(cfg, 'dist_params'):
        cfg.dist_params = dict(backend='nccl')

    init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Config: {cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)

    cfg.seed = seed
    meta['seed'] = seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))

    model = build_model(cfg.model)

    datasets = [build_dataset(cfg.data.train)]

    cfg.workflow = cfg.get('workflow', [('train', 1)])
    assert len(cfg.workflow) == 1
    if cfg.checkpoint_config is not None:
        # save pyskl version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            pyskl_version=__version__ + get_git_hash(digits=7),
            tag=['nturgb+d graph changed', 'scale A > 1 in training, unchanged in testing'],
            config=cfg.pretty_text)

    test_option = dict(test_last=args.test_last, test_best=args.test_best)

    # We need to write the keys into memcache
    so_keys, cli = set(), None
    if rank == 0:
        mc_list = cfg.get('mc_list', None)
        if mc_list is not None:
            mc_cfg = cfg.get('mc_cfg', ('localhost', 11211))
            assert isinstance(mc_cfg, tuple) and mc_cfg[0] == 'localhost'
            if not test_port(mc_cfg[0], mc_cfg[1]):
                mc_on(port=mc_cfg[1], launcher=args.launcher)
            retry = 3
            while not test_port(mc_cfg[0], mc_cfg[1]) and retry > 0:
                time.sleep(5)
                retry -= 1
            assert retry >= 0, 'Failed to launch memcached. '
            assert isinstance(mc_list, list)
            from pymemcache.client.base import Client
            from pymemcache import serde
            cli = Client(mc_cfg, serde=serde.pickle_serde)
            for data_file in mc_list:
                assert osp.exists(data_file)
                kv_dict = load(data_file)
                if isinstance(kv_dict, list):
                    assert ('frame_dir' in kv_dict[0]) != ('filename' in kv_dict[0])
                    key = 'frame_dir' if 'frame_dir' in kv_dict[0] else 'filename'
                    kv_dict = {x[key]: x for x in kv_dict}
                for k, v in kv_dict.items():
                    assert k not in so_keys
                    cli.set(k, v)
                    so_keys.add(k)
    dist.barrier()

    train_model(
        model,
        datasets,
        cfg,
        validate=args.validate,
        test=test_option,
        timestamp=timestamp,
        meta=meta)

    dist.barrier()
    if rank == 0 and cli is not None:
        for k in so_keys:
            cli.delete(k)
        mc_off()


if __name__ == '__main__':
    main()
