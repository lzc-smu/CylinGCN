from .yacs import CfgNode as CN
import argparse
import os


cfg = CN()

# model
cfg.model = ''
cfg.model_dir = 'data/checkpoints'

# network
cfg.network = ''

# network heads
cfg.heads = CN()

# task
cfg.task = ''

# gpus
cfg.gpus = 0


# if load the pretrained network
cfg.resume = True


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.epoch = 200
cfg.train.num_workers = 1

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.scheduler = ''
cfg.train.milestones = [80, 120, 200, 240]
cfg.train.gamma = 0.5

cfg.train.batch_size = 4
cfg.train.image_size = 1

# val
cfg.val = CN()
cfg.val.batch_size = 1
cfg.val.epoch = -1
cfg.val.image_size = 1


# test
cfg.test = CN()
cfg.test.batch_size = 1
cfg.test.image_size = 1
cfg.test.radius_len = 0.7

# recorder
cfg.record_dir = 'data/record'

# result
cfg.result_dir = 'data/result'


cfg.save_ep = 5
cfg.eval_ep = 5




def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(str(cfg.gpus))

    # assign the network head conv
    cfg.head_conv = 64 if 'res' in cfg.network else 256

    cfg.model_dir = os.path.join(cfg.model_dir, cfg.task)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task)


def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--det', type=str, default='')
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
cfg = make_cfg(args)
