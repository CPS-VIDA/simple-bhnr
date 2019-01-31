import torch
import numpy as np
import random

import gym


def set_global_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        # is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        # if is_atari:
        #     env = make_atari(env_id)
        env.seed(seed + rank)
        # if is_atari:
        #     env = wrap_deepmind(env)
        #     env = WrapPyTorch(env)
        return env

    return _thunk
