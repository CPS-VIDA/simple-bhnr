import copy
import os

import gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from safe_rl.core.agent import BaseAgent
from safe_rl.observers.checkpoints import EpisodicCheckpointSaver
from safe_rl.utils.general import set_global_seed

import pandas as pd
from datetime import datetime

def trial_runner(agent_fn, conf: dict):
    env_id = conf['env_id']
    name = conf['name']
    n_trials = conf['n_trials']
    n_episodes = conf['n_episodes']
    n_eval_episodes = conf['n_eval_episodes']
    render = conf['render']
    save_dir = conf['save_dir']
    backup_dir = conf['backup_dir']
    backup_interval = conf['backup_interval']
    device = conf['device']
    seed_fn = conf.get('seed_fn', lambda cur_seed: 0 if cur_seed is None else cur_seed + 1)

    for trial in range(n_trials):
        conf['seed'] = seed_fn(conf.get('seed'))
        trial_num_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        trial_dir = os.path.join(save_dir, name)
        checkpoint_dir = os.path.join(backup_dir, name, trial_num_str)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(trial_dir, exist_ok=True)

        agent = agent_fn(conf)
        agent.attach(EpisodicCheckpointSaver(checkpoint_dir, interval=backup_interval))
        durations, rewards = agent.run_training(n_episodes, render=render)

        trial_num_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        monitor_csv = os.path.join(trial_dir, trial_num_str + '.monitor.csv')
        df = pd.DataFrame({
            'dur': durations,
            'rew': rewards,
        })
        df.to_csv(monitor_csv, header=False)



        
