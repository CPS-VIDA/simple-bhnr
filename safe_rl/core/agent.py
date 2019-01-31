from abc import ABC, abstractmethod
from collections import deque

import torch
import gym
from safe_rl.utils.general import set_global_seed

from gym.spaces import Box

class BaseAgent(ABC):
    """Abstract Agent class"""

    default_hyperparams = dict()

    def __init__(self, env_id, hyperparams, **kwargs):
        self.env_id = env_id
        self.hyp = self.default_hyperparams
        self.hyp.update(hyperparams)
        self.observers = deque()

        tmp_dev = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = kwargs.get('device', tmp_dev)
        self._seed = kwargs.get('seed')

        self._init_env()
        self.observation_space = self.env.observation_space # type: Box
        self.action_space = self.env.action_space

    def _init_env(self):
        self.env = gym.make(self.env_id)
        self.env.seed(self._seed)

    @abstractmethod
    def run_training(self, n_episodes, render=False):
        pass

    @abstractmethod
    def run_eval(self, n_episodes, render=False):
        pass

    @abstractmethod
    def act(self, state_vec):
        pass

    @abstractmethod
    def observe(self, *args):
        pass

    @abstractmethod
    def learn(self, *args):
        pass

    @abstractmethod
    def to(self, torch_device):
        pass

    def broadcast(self, msg):
        for obs in self.observers:
            obs.notify(msg)

    def attach(self, observer):
        observer.attach(self)
        self.observers.append(observer)

    def _get_tensor(self, x, dtype=torch.float):
        return torch.tensor(x, device=self.device, dtype=dtype)

    @abstractmethod
    def load_net(self, filepath):
        pass

    @abstractmethod
    def save_net(self, filepath):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def seed(self, seed=None):
        self.env.seed(seed)

