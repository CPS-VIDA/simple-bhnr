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
from safe_rl.utils.general import set_global_seed
from temporal_logic.signal_tl.semantics import (EfficientRobustnessMonitor,
                                                FilteringMonitor)

SIGNALS = ('x', 'x_dot', 'theta', 'theta_dot')
x, x_dot, theta, theta_dot = stl.signals(SIGNALS)
SPEC = stl.G(stl.F(x_dot < abs(0.01)) & (abs(theta) < 5)
             & (abs(x) < 0.5) & stl.F(abs(theta) < 1))


def parse_args():
    parser = argparse.ArgumentParser(description='Eval DQN')
    parser.add_argument('--env-id', required=True, help='Gym env id')
    parser.add_argument('--name', required=True, help='Name of the current experiment')
    parser.add_argument('--n-eval-episodes', default=100, type=int, help='Number of episodes to evaluate the policy')
    parser.add_argument('--render', action='store_true', default=False, help='Render the env?')
    parser.add_argument('--save-dir', required=True, help='Base directory for all your data')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--load', )
    parser.add_argument('--device', type=str, default='cpu')

    return parser.parse_args()

def evaluate(args):
    net=nn.Sequential(
        nn.Linear(4, 24), nn.ReLU(),
        nn.Linear(24, 24), nn.ReLU(),
        nn.Linear(24, 2)
    )
    hyperparams=dict(
        gamma=0.95,

        lr=0.001,

        batch_size=32,

        double_q=True,
        target_net=False,
        target_update=15,

        memory_size=2000,
        memory_type='uniform',  # or per

        epsilon_greedy=True,
        epsilon=0.0,
        epsilon_min=0.1,
        epsilon_decay=0.95,
    )
    agent = DQN(
        args.env_id, hyperparams, net, 
        device=torch.device(args.device),
        seed=args.seed
    )
    agent.eval()
    agent.load_net(args.load)

    monitor = EfficientRobustnessMonitor(SPEC, SIGNALS)

    output_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(output_dir, exist_ok=True)
    for ep in range(args.n_eval_episodes):
        states, rewards = agent.eval_episode(
            render=args.render,
            seed=args.seed,
            load=args.load)
        states = np.asarray(states)
        rewards = np.reshape(rewards, (-1, 1))
        robustness = np.reshape(monitor(states), (-1, 1))
        df = pd.DataFrame(np.concatenate((states, rewards, robustness), axis=1))
        eval_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        df.to_csv(os.path.join(output_dir, eval_str + '.eval.csv'))
    
if __name__ == "__main__":
    evaluate(parse_args())
