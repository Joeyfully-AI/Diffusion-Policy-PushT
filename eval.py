"""
Usage:
python eval.py --checkpoint data/outputs/.../checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
import os
import logging
import warnings
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ.setdefault('WANDB_SILENT', 'true')
os.environ.setdefault('PYTHONWARNINGS', 'ignore')
logging.getLogger('wandb').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message=r'Gym has been unmaintained since 2022.*')
warnings.filterwarnings('ignore', message=r'The `format` argument was not provided, defaulting to `gif`\..*')

import os
import pathlib
import shutil
import click
import hydra
import torch
import dill
import json
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# Edit these values directly when you want to change evaluation behavior.
EVAL_N_TRAIN = 0
EVAL_N_TRAIN_VIS = 0
EVAL_N_TEST = 40
EVAL_N_TEST_VIS = 40

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    if EVAL_N_TRAIN is not None:
        OmegaConf.update(cfg, 'task.env_runner.n_train', EVAL_N_TRAIN, force_add=True)
    if EVAL_N_TRAIN_VIS is not None:
        OmegaConf.update(cfg, 'task.env_runner.n_train_vis', EVAL_N_TRAIN_VIS, force_add=True)
    if EVAL_N_TEST is not None:
        OmegaConf.update(cfg, 'task.env_runner.n_test', EVAL_N_TEST, force_add=True)
    if EVAL_N_TEST_VIS is not None:
        OmegaConf.update(cfg, 'task.env_runner.n_test_vis', EVAL_N_TEST_VIS, force_add=True)

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    video_dir_name = pathlib.Path(checkpoint).stem
    runner_log = env_runner.run(policy, video_dir_name=video_dir_name)

    json_log = dict()
    for key, value in runner_log.items():
        if hasattr(value, '_path'):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'log', 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
