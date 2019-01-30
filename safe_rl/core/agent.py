from abc import ABC, abstractmethod
from collections import deque

import torch
import gym

class BaseAgent(ABC):
    """Abstract Agent class"""

    default_hyperparams = dict()

    def __init__(self, env_id, hyperparams):
        self.env_id = env_id
        self._init_env()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.hyp = self.default_hyperparams
        self.hyp.update(hyperparams)
        self.observers = deque()
        self.device = torch.device('cpu')

    def _init_env(self):
        self.env = gym.make(self.env_id)

    @abstractmethod
    def train(self, n_episodes, render=False):
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

    def _get_tensor(self, x, dtype=torch.float) -> torch.Tensor:
        return torch.tensor(x, device=self.device, dtype=dtype)
