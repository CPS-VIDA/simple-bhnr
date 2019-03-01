import torch
import numpy as np
import random

import gym

from safe_rl.envs import SubprocVecEnv, DummyVecEnv
import vrep_gym

import atexit



def set_global_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        if seed is not None:
            env.seed(int(seed) + rank)
        else:
            env.seed(seed)
        return env

    return _thunk


def make_vec_env(env_id, seed, num_envs):
    def _make_env(_rank):
        def _thunk():
            env = gym.make(env_id)
            if seed is not None:
                env.seed(int(seed) + _rank)
            else:
                env.seed(seed)
            return env
        return _thunk

    envs = [_make_env(i) for i in range(num_envs)]
    if len(envs) > 1:
        return SubprocVecEnv(envs)
    return DummyVecEnv(envs)
