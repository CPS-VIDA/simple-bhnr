from abc import ABC, abstractmethod
from collections import deque

import gym
import numpy as np
import torch

from safe_rl.utils.general import set_global_seed


class BaseAgent(ABC):
    """Abstract Agent class"""

    default_hyperparams = dict()

    def __init__(self, env_id, hyperparams, **kwargs):
        self.env_id = env_id
        self._hyp = self.default_hyperparams
        self._hyp.update(hyperparams)
        self.observers = deque()

        tmp_dev = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._device = kwargs.get('device', tmp_dev)
        self._seed = kwargs.get('seed')

        self._init_env()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _init_env(self):
        self.env = gym.make(self.env_id)
        self.env.seed(self._seed)

    @property
    def hyp(self):
        """Return the Hyperparameter dict for agent

        Returns
        -------
        dict :
            Hyperparam dict for agent
        """

        return self._hyp

    @property
    def device(self):
        """Return the default torch.device for this a agent

        Returns
        -------
        torch.device
            The default PyTorch device to run this model on
        """

        return self._device

    @abstractmethod
    def run_training(self, n_episodes, render=False):
        """Run training for n episodes

        Run agent-specific training for n-episodes

        Parameters
        ----------
        n_episodes : int
            Number of episodes to run the agent training for
        render : bool, optional
            Render the graphical output to the screen (the default is False,
            which doesn't render the simulation)
        """
        pass

    @abstractmethod
    def eval_episode(self, render=False, seed=None, load=''):
        """Run an episode for evaluating the algorithm

        Parameters
        ----------
        render : bool, optional
            Render the simulation to the screen (the default is False)
        seed : int or None, optional
            Random seed for the episode (the default is None)
        load : str, optional
            Load policy parameters from path (the default is '', which means
            random policy)

        Returns
        -------
        dict :
            Data pertaining to the eval episode, typically states in the
            episode.
        """
        pass

    @abstractmethod
    def act(self, state_vec, deterministic=False):
        """Run the policy function

        Output a list of actions based on the current policy, given states

        Parameters
        ----------
        state_vec : numpy.ndarray or torch.Tensor
            Vector of states from the environment
        deterministic : bool, optional
            Choose a deterministic action (the default is False, which chooses
            an epsilon-greedy/distribution-based policy).

        Returns
        -------
        numpy.ndarray :
            A list of actions for each state inputted in the vector
        """
        return np.empty((1,))

    @abstractmethod
    def observe(self, *args):
        """Provide the agent with the new transitions so 

        Provide new transitions so that it can remember/replay
        """

        return

    @abstractmethod
    def learn(self, *args):
        """Optimize agent parameter
        """
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

    def set_global_seed(self, seed=None):
        set_global_seed(seed)
