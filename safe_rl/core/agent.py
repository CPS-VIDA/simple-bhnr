from abc import ABC, abstractmethod
from collections import deque

import torch
import gym


class BaseAgent(ABC):
    """Abstract Agent class"""

    default_hyperparams = dict()

    def __init__(self, env_id, hyperparams, **kwargs):
        self.env_id = env_id
        self._init_env()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.hyp = self.default_hyperparams
        self.hyp.update(hyperparams)
        self.observers = deque()

        tmp_dev = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = kwargs.get('device', tmp_dev)

    def _init_env(self):
        self.env = gym.make(self.env_id)

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
    def observe(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def learn(self):
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

