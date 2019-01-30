import copy
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
from safe_rl.core.agent import BaseAgent
from safe_rl.core.observers import Event
from safe_rl.memory import (PrioritizedMemory, QTransition, Transition,
                            UniformReplay)
from safe_rl.observers import EpsilonGreedyUpdater, TargetUpdater


class DQN(BaseAgent):
    default_hyperparams = dict(
        gamma=0.95,

        lr=0.001,

        batch_size=32,

        double_q=False,
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

        :param env: OpenAI Gym-compatible environment id
        :type env: str
        :param hyperparams: Hyper-parameters for the agent
        :type hyperparams: dict
        :param model: NN architecture for DQN
        :type model: nn.Module
        """
        super(DQN, self).__init__(env, hyperparams)

        self.gamma = self.hyp['gamma']
        self.batch_size = self.hyp['batch_size']

        self.epsilon_greedy = self.hyp['epsilon_greedy']
        self.epsilon = self.hyp['epsilon']
        self.epsilon_min = self.hyp['epsilon_min']
        self.epsilon_decay = self.hyp['epsilon_decay']
        if self.epsilon_greedy:
            self.attach(EpsilonGreedyUpdater(
                self.epsilon_min, self.epsilon_decay))

        self.lr = self.hyp['lr']

        self.policy_net = copy.deepcopy(model)

        if self.hyp['double_q']:
            self.hyp['target_net'] = True

        if self.hyp['target_net']:
            self.target_net = copy.deepcopy(self.policy_net)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.target_update = self.hyp['target_update']
            self.target_update_count = 0
            self.attach(TargetUpdater(self.target_update))

        mem_type = self.hyp['memory_type']
        mem_size = self.hyp['memory_size']
        if mem_type == 'uniform':
            self.memory = UniformReplay(mem_size)
        elif mem_type == 'per':
            self.memory = PrioritizedMemory(mem_size)
        else:
            raise ValueError('Incompatible memory type: {}'.format(mem_type))

        self.loss_fn = kwargs.get('loss_fn', Fn.mse_loss)
        optim_fn = kwargs.get(
            'optimizer_fn', lambda params: optim.Adam(params))
        self.optimizer = optim_fn(self.policy_net.parameters())

        device = kwargs.get('device', 'cpu')

        self.episode_count = 0
        self.episode_rewards = deque()
        self.episode_durations = deque()

    def observe(self, state, action, reward, next_state, done):
        self.memory.push(Transition(state, action, reward, next_state, done))

    def act(self, state):
        if self.epsilon_greedy and np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        with torch.no_grad():
            s = self._get_tensor(state).unsqueeze(0)
            a = self.policy_net(s)
            return torch.argmax(a[0]).item()

    def _recall(self, batch_size):
        sample = self.memory.sample(batch_size)
        batch = Transition(*zip(*sample))
        state_batch = self._get_tensor(batch.state)
        action_batch = self._get_tensor(batch.action, dtype=torch.long)
        reward_batch = self._get_tensor(batch.reward)
        next_state_batch = self._get_tensor(batch.next_state)
        final_states = self._get_tensor(batch.done, dtype=torch.uint8)

        return Transition(state_batch, action_batch, reward_batch, next_state_batch, final_states)

    def _q_update(self, transition):
        states, actions, rewards, next_states, final_states = transition
        if self.hyp['target_net']:
            if self.hyp['double_q']:
                policy_next_q = self.policy_net(next_states)
                argmax_policy_q = policy_next_q.max(1)[1].unsqueeze(1)
                q_max = self.target_net(next_states) \
                    .gather(1, argmax_policy_q) \
                    .squeeze(1)
            else:
                q_max = self.target_net(next_states).max(1)[0].detach()
        else:
            q_max = self.policy_net(next_states).max(1)[0].detach()

        target_value = rewards + (self.gamma * q_max)
        target_vector = torch.where(final_states, rewards, target_value)
        return target_vector.unsqueeze(1).detach()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None, None
        tensor_batch = self._recall(self.batch_size)
        target_vector = self._q_update(tensor_batch)

        state_batch = tensor_batch.state
        action_batch = tensor_batch.action
        outputs = self.policy_net(state_batch).gather(
            1, action_batch.unsqueeze(1))

        return outputs, target_vector

    def learn(self, predicted, target):
        if predicted is None or target is None:
            return
        self.broadcast(Event.BEGIN_LEARN)

        if self.hyp['memory_type'] == 'per':
            err = torch.abs(predicted - target).detach()
            idxs, weights = self.memory.last_idx_weights
            for i, idx in enumerate(idxs):
                self.memory.update(idx, err[i].item())

        self.optimizer.zero_grad()
        loss = self.loss_fn(predicted, target)
        loss.backward()
        self.optimizer.step()
        self.broadcast(Event.END_LEARN)

    def to(self, torch_device):
        self.device = torch_device
        self.policy_net.to(torch_device)
        if self.target_net:
            self.target_net.to(torch_device)

    def train(self, n_episodes, render=False):
        self.broadcast(Event.BEGIN_TRAINING)
        for ep in range(n_episodes):
            self.train_episode(render)
            duration, total_reward = self.episode_durations[-1], self.episode_rewards[-1]
            print('[{:{width}d}/{:d}] Duration: {:6d}, Score: {:6.2f}, Epsilon: {:.2f}'.format(
                ep,
                n_episodes,
                duration,
                total_reward,
                self.epsilon,
                width=len(str(n_episodes))))
        self.broadcast(Event.END_TRAINING)
        return self.episode_durations, self.episode_rewards

    def train_episode(self, render=False):
        self.broadcast(Event.BEGIN_EPISODE)
        total_reward = 0
        state = self.env.reset()

        t = 0
        done = False
        while not done:
            if render:
                self.env.render()
            action = self.act(state)
            obs, rew, done, _ = self.env.step(action)
            self.observe(state, action, rew, obs, done)
            self.broadcast(Event.STEP)
            total_reward += rew
            t += 1
            state = obs
            output, target = self.replay()
            self.learn(output, target)
        self.episode_count += 1
        self.episode_durations.append(t)
        self.episode_rewards.append(total_reward)
        self.broadcast(Event.END_EPISODE)
