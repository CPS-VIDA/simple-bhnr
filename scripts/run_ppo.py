import argparse
import copy
import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn

import temporal_logic.signal_tl as stl
from safe_rl.agents.ppo import PPO
from safe_rl.experiments.trial import trial_runner
from safe_rl.observers.checkpoints import EpisodicCheckpointSaver
from safe_rl.observers.multi_proc_stl import MultiProcSTLRewarder
from safe_rl.utils.general import set_global_seed
from temporal_logic.signal_tl.semantics import (EfficientRobustnessMonitor,
                                                FilteringMonitor)

from safe_rl.specs import get_spec

import logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Run PPO')
    parser.add_argument('--env-id', required=True, help='Gym env id')
    parser.add_argument('--name', required=True,
                        help='Name of the current experiment')
    # parser.add_argument('--n-trials', default=8, type=int, help='Number of tr')
    parser.add_argument('--maximum-steps', default=100000,
                        type=int, help='Total number of steps desired')
    parser.add_argument('--n-eval-episodes', default=100, type=int,
                        help='Number of episodes to evaluate the policy')
    parser.add_argument('--render', action='store_true',
                        default=False, help='Render the env?')
    parser.add_argument('--save-dir', default='data',
                        help='Base directory for all your data')
    parser.add_argument('--backup-dir', default='backups',
                        help='Base dir for all backup data')
    parser.add_argument('--backup-interval', default=100, type=int)
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--gamma', type=float, default=0.95)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.99)

    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)

    parser.add_argument('--n-steps', type=int, default=32)
    parser.add_argument('--n-workers', type=int, default=4)

    parser.add_argument('--use-gae', action='store_true', default=False)

    parser.add_argument('--clipping', type=float, default=0.2)
    parser.add_argument('--use-clipped', action='store_true', default=True)
    parser.add_argument('--ppo-epoch', type=int, default=4)
    parser.add_argument('--n-minibatch', type=int, default=32)

    parser.add_argument('--seed', default=None)

    parser.add_argument('--use-stl', action='store_true', default=False)

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--load', default=None)

    args = parser.parse_args()
    if args.eval and args.load is None:
        parser.error('--eval mode requires providing --load state dict')

    config = dict(
        env_id=args.env_id,
        name=args.name,
        maximum_steps=args.maximum_steps,
        n_eval_episodes=args.n_eval_episodes,
        render=args.render,
        save_dir=args.save_dir,
        backup_dir=args.backup_dir,
        backup_interval=args.backup_interval,
        device=torch.device(args.device),
        hyperparams=dict(
            gamma=args.gamma,

            lr=args.lr,
            epsilon=args.epsilon,
            alpha=args.alpha,

            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,

            n_steps=args.n_steps,
            n_workers=args.n_workers,

            use_gae=args.use_gae,

            clipping=args.clipping,
            use_clipped_value_loss=args.use_clipped,
            ppo_epochs=args.ppo_epoch,
            n_minibatch=args.n_minibatch,
        ),
        seed=args.seed,
        use_stl=args.use_stl,
        eval=args.eval,
        load=args.load,
    )
    return config


def gen_agent(conf):
    conf = conf.copy()
    env_id = conf.pop('env_id')
    # net = copy.deepcopy(conf.pop('net'))
    hyp = conf.pop('hyperparams')
    agent = PPO(env_id, hyp, **conf)
    if conf['use_stl']:
        spec, signals, monitor = get_spec(env_id)
        monitor = monitor(spec, signals)
        n_steps = hyp['n_steps']
        n_workers = hyp['n_workers']
        agent.attach(MultiProcSTLRewarder(n_steps, n_workers, monitor))
    return agent


def run_training(conf):
    env_id = conf['env_id']
    name = conf['name']
    maximum_steps = conf['maximum_steps']
    n_eval_episodes = conf['n_eval_episodes']
    render = conf['render']
    save_dir = conf['save_dir']
    backup_dir = conf['backup_dir']
    backup_interval = conf['backup_interval']
    device = conf['device']
    seed = None if conf['seed'] is None else int(conf['seed'])

    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    trial_dir = os.path.join(save_dir, name)
    check_dir = os.path.join(backup_dir, name, now)
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)

    agent = gen_agent(conf)
    agent.attach(EpisodicCheckpointSaver(check_dir, interval=backup_interval))
    set_global_seed(seed)

    data = agent.run_training(maximum_steps, render=render)
    if data is not None:
        df = pd.DataFrame(data)
        now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        mon_csv = os.path.join(trial_dir, now + '.monitor.csv')
        df.to_csv(mon_csv, header=False)


def run_eval(conf):
    env_id = conf['env_id']
    name = conf['name']
    n_eval_episodes = conf['n_eval_episodes']
    render = conf['render']
    save_dir = conf['save_dir']
    backup_dir = conf['backup_dir']
    backup_interval = conf['backup_interval']
    device = conf['device']
    seed = conf['seed']
    load = conf['load']

    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    trial_dir = os.path.join(save_dir, name)
    check_dir = os.path.join(backup_dir, name, now)
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)

    agent = gen_agent(conf)
    agent.attach(EpisodicCheckpointSaver(check_dir, interval=backup_interval))
    set_global_seed(seed)

    data = agent.eval_episode(render=render, seed=seed, load=load)
    if data is not None:
        df = pd.DataFrame(data)
        now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        mon_csv = os.path.join(trial_dir, now + '.eval.csv')
        df.to_csv(mon_csv, header=False)


if __name__ == "__main__":
    conf = parse_args()
    if conf['eval']:
        run_eval(conf)
    run_training(conf)
