from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract Agent class"""

    default_hyperparams = dict()

    def __init__(self, env, hyperparams):
        self.env = env
        self.hyp = self.default_hyperparams
        self.hyp.update(hyperparams)

    @abstractmethod
    def train(self, n_episodes):
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
