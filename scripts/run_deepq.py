import argparse
import copy
import os
import random
from datetime import datetime

import gym
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn

import temporal_logic.signal_tl as stl
from safe_rl.agents.deepq import DQN
from safe_rl.observers.checkpoints import EpisodicCheckpointSaver
from safe_rl.observers.stl_rewards import STLRewarder
from safe_rl.utils.general import set_global_seed
from temporal_logic.signal_tl.semantics import (EfficientRobustnessMonitor,
                                                FilteringMonitor)

SIGNALS = ('x', 'x_dot', 'theta', 'theta_dot')
x, x_dot, theta, theta_dot = stl.signals(SIGNALS)
SPEC = stl.G(stl.F(x_dot < abs(0.01)) & (abs(theta) < 5)
             & (abs(x) < 0.5) & stl.F(abs(theta) < 1))


def parse_args():
    parser = argparse.ArgumentParser(description='Run DQN')
    parser.add_argument('--env-id', required=True, help='Gym env id')
    parser.add_argument('--name', required=True,
                        help='Name of the current experiment')
    parser.add_argument('--n-episodes', default=2000,
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

    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--epsilon-greedy', action='store_false', default=True)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon-min', type=float, default=0.1)
    parser.add_argument('--epsilon-decay', type=float, default=0.95)

    parser.add_argument('--memory-size', type=int, default=2000)
    parser.add_argument('--memory-type', type=str,
                        default='uniform', choices=['uniform', 'per'])

    parser.add_argument('--target-net', action='store_true', default=False)
    parser.add_argument('--target-update', type=int, default=15)
    parser.add_argument('--double-q', action='store_true', default=False)

    parser.add_argument('--psiglen', type=int, default=15)

    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--use-stl', action='store_true', default=False)

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--load', default=None)

    args = parser.parse_args()
    if args.eval and args.load is None:
        parser.error('--eval mode requires providing --load state dict')

    config = dict(
        env_id=args.env_id,
        name=args.name,
        n_episodes=args.n_episodes,
        n_eval_episodes=args.n_eval_episodes,
        render=args.render,
        save_dir=args.save_dir,
        backup_dir=args.backup_dir,
        backup_interval=args.backup_interval,
        device=torch.device(args.device),
        hyperparams=dict(
            gamma=args.gamma,
            lr=args.lr,
            batch_size=args.batch_size,

            epsilon_greedy=args.epsilon_greedy,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,

            memory_size=args.memory_size,
            memory_type=args.memory_type,
            target_net=args.target_net,
            target_update=args.target_update,
            double_q=args.double_q,

            psiglen=args.psiglen,
        ),
        seed=args.seed,
        use_stl=args.use_stl,
        eval=args.eval,
        load=args.load,
    )
    return config


def gen_agent(conf):
    net = nn.Sequential(
        nn.Linear(4, 24), nn.ReLU(),
        nn.Linear(24, 24), nn.ReLU(),
        nn.Linear(24, 2)
    )

    conf = conf.copy()
    env_id = conf.pop('env_id')
    hyp = conf.pop('hyperparams')
    agent = DQN(env_id, hyp, net, **conf)
    if conf['use_stl']:
        monitor = FilteringMonitor(SPEC, SIGNALS)
        agent.attach(STLRewarder(monitor, hyp['psiglen']))
    return agent


def run_training(conf):
    env_id = conf['env_id']
    name = conf['name']
    n_episodes = conf['n_episodes']
    render = conf['render']
    save_dir = conf['save_dir']
    backup_dir = conf['backup_dir']
    backup_interval = conf['backup_interval']
    device = conf['device']
    seed = conf['seed']

    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    trial_dir = os.path.join(save_dir, name)
    check_dir = os.path.join(backup_dir, name, now)
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)

    agent = gen_agent(conf)
    agent.attach(EpisodicCheckpointSaver(check_dir, interval=backup_interval))
    set_global_seed(seed)

    data = agent.run_training(n_episodes, render=render)
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
    config = parse_args()
    if config['eval']:
        run_eval(config)
    run_training(config)
