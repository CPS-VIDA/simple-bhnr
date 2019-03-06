from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from safe_rl.models.distributions import Categorical, DiagGaussian

from gym.spaces import Discrete, Box


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


class ActorCritic(nn.Module):
    def __init__(self, n_inputs, action_space):
        super(ActorCritic, self).__init__()

        self.linear1 = nn.Linear(n_inputs, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)

        if isinstance(action_space, Discrete):
            n_actions = action_space.n
            self.dist = Categorical(64, n_actions)
        elif isinstance(action_space, Box):
            n_actions = action_space.shape[0]
            self.dist = DiagGaussian(64, n_actions)
        else:
            raise NotImplementedError('Unsupported action space')

        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)
        self.apply(init_weights)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)
        x = F.relu(x)

        return self.critic(x), x

    def evaluate_actions(self, inputs, actions):
        value, x = self(inputs)
        dist = self.dist(x)  # type: torch.distributions.Independent
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy

    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        dist = self.dist(x)  # type: torch.distributions.Independent
        action = dist.sample()
        action_log_probs = dist.log_prob(action)
        return value, action, action_log_probs
