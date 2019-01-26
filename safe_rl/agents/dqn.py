from safe_rl.core.agent import BaseAgent
from safe_rl.utils.memory import UniformReplay, PrioritizedMemory

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Fn
import gym


class DQN(BaseAgent):
    default_hyperparams = dict(
        gamma=0.95,

        lr=0.001,

        batch_size=32,

        target_net=False,
        target_update=15,

        memory_size=2000,
        memory_type='uniform',  # or per

        epsilon_greedy=True,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.95,
    )

    def __init__(self, env, hyperparams, model, **kwargs):
        """Deep Q Networks and its variants.
        [1] V. Mnih et al., “Playing Atari with Deep Reinforcement Learning,” arXiv:1312.5602 [cs], Dec. 2013.
        [2] H. van Hasselt, A. Guez, and D. Silver, “Deep Reinforcement Learning with Double Q-learning,”
            arXiv:1509.06461 [cs], Sep. 2015.

        :param env: OpenAI Gym-compatible environment
        :type env: gym.Env
        :param hyperparams: Hyper-parameters for the agent
        :type hyperparams: dict
        :param model: NN architecture for DQN
        :type model: nn.Module
        """
        super(DQN, self).__init__(env, hyperparams)

        self.gamma = self.hyp['gamma']
        self.gamma = self.hyp['gamma']

        # TODO: Epsilon greedy explorer
        self.epsilon = self.hyp['epsilon']
        self.epsilon_min = self.hyp['epsilon_min']
        self.epsilon_decay = self.hyp['epsilon_decay']
        self.lr = self.hyp['lr']

        self.policy_net = copy.deepcopy(model)

        if self.hyp['target_net']:
            self.target_net = copy.deepcopy(self.policy_net)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.target_update = self.hyp['target_update']
            self.target_update_count = 0

        mem_type = self.hyp['memory_type']
        mem_size = self.hyp['memory_size']
        if mem_type == 'uniform':
            self.memory = UniformReplay(mem_size)
        elif mem_type == 'per':
            self.memory = PrioritizedMemory(mem_size)
        else:
            raise ValueError('Incompatible memory type: {}'.format(mem_type))

        self.loss_fn = kwargs.get('loss_fn', Fn.mse_loss)
        optim_fn = kwargs.get('optimizer_fn', lambda params: optim.Adam(params))
        self.optimizer = optim_fn(self.policy_net.parameters())


