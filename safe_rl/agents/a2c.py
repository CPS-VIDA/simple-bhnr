import copy
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from safe_rl.core.agent import BaseAgent

from safe_rl.utils.general import make_env, make_vec_env
from safe_rl.core.observers import Event
from safe_rl.envs import SubprocVecEnv, DummyVecEnv
from safe_rl.models.ac import ActorCritic
from safe_rl.core.memory import ACTransition
from safe_rl.memory.rollout import MultiProcRolloutMemory


import logging
log = logging.getLogger(__name__)


class A2C(BaseAgent):
    default_hyperparams = dict(
        gamma=0.95,

        lr=0.001,
        epsilon=1e-5,
        alpha=0.99,

        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,

        n_steps=5,
        n_workers=16,

        use_gae=False,
        tau=0.95,
    )

    def __init__(self, env_id, hyperparams, **kwargs):
        super(A2C, self).__init__(env_id, hyperparams, **kwargs)

        self.gamma = self.hyp['gamma']
        self.lr = self.hyp['lr']
        self.epsilon = self.hyp['epsilon']
        self.alpha = self.hyp['alpha']
        self.value_loss_coef = self.hyp['value_loss_coef']
        self.entropy_coef = self.hyp['entropy_coef']
        self.max_grad_norm = self.hyp['max_grad_norm']
        self.n_steps = self.hyp['n_steps']
        self.n_workers = self.hyp['n_workers']

        self.use_gae = self.hyp['use_gae']
        self.tau = self.hyp['tau']

        # assert isinstance(model, ActorCritic), 'The given nn.Module is not an ActorCritic Model'
        model = ActorCritic(self.observation_space.shape[0], self.action_space)
        self.policy_net = copy.deepcopy(model)
        optim_fn = self.hyp.get('optim_fn',
                                lambda params: optim.RMSprop(params, lr=self.lr, alpha=self.alpha, eps=self.epsilon))
        self.optimizer = optim_fn(self.policy_net.parameters())

        self.memory = MultiProcRolloutMemory(self.n_steps, self.n_workers)

        self.mean_step_rewards = deque()
        self.episode_rewards = np.zeros(self.n_workers)

        self.to(self.device)

    def _init_env(self):
        self.env = make_vec_env(self.env_id, self._seed, self.hyp['n_workers'])

    def run_training(self, total_steps, render=False):
        num_updates = total_steps // self.n_steps // self.n_workers
        log.info('Number of updates to run: {}'.format(num_updates))
        self.broadcast(Event.BEGIN_TRAINING)
        states = self.env.reset()
        for epoch in range(num_updates):
            log.info('Begin rollout simulation #{}'.format(epoch))
            self.broadcast(Event.BEGIN_EPISODE)
            states = self.rollout(states, render)
            self.process_rollout()
            self.memory.clear_until_last()
            self.broadcast(Event.END_EPISODE)
        self.broadcast(Event.END_TRAINING)

        max_steps = total_steps // self.n_workers
        ret = np.array(self.mean_step_rewards)[:max_steps]
        log.debug('Number of rollouts: {}'.format(len(ret)))
        log.debug('Number of rollouts: {} ^'.format(max_steps))
        max_steps = min(max_steps, len(ret))
        return dict(frames=np.arange(max_steps), mean_returns=ret)

    def act(self, state_vec, deterministic=False):
        state_vec = self._get_tensor(state_vec)
        value, action, action_log_prob = self.policy_net.act(state_vec)
        return (
            value.detach().cpu().numpy(),
            action.detach().cpu().numpy(),
            action_log_prob.detach().cpu().numpy()
        )

    def observe(self, state, action, reward, next_state, done, action_log_prob, value_pred):
        transition_list = ACTransition(
            state, action, reward, next_state, done, action_log_prob, value_pred, reward)
        transitions = [ACTransition(*t) for t in zip(*transition_list)]
        self.memory.push(transitions)

    def rollout(self, states, render=False):
        for step in range(self.n_steps):
            if render:
                self.env.render()
            val, action, action_log_prob = self.act(states)
            obs, rew, done, _ = self.env.step(action)
            self.observe(states, action, rew, obs, done, action_log_prob, val)
            states = obs
            self.broadcast(Event.STEP)
        next_vals = self.policy_net(self._get_tensor(states))[
            0].detach().cpu().numpy()
        self.broadcast(Event.SYNC)
        self.memory.compute_returns(
            next_vals, self.use_gae, self.gamma, self.tau)
        return states

    def process_rollout(self):
        unroll_pre = ACTransition(*zip(*self.memory.sample(size=-1)))
        next_state_batch = torch.cat(
            tuple(self._get_tensor(unroll_pre.next_state)))
        action_batch = torch.cat(tuple(self._get_tensor(unroll_pre.action)))
        returns_batch = torch.cat(
            tuple(self._get_tensor(unroll_pre.returns))).squeeze()

        msr = np.array(unroll_pre.reward).mean(axis=0)
        sum_ret = np.array(unroll_pre.returns).mean(axis=0).sum()
        log.info('Mean Step Return: {}'.format(sum_ret))
        self.mean_step_rewards.extend(msr)
        self.episode_rewards += np.array(unroll_pre.reward).sum(axis=1)

        vals, action_log_probs, dist_entropy = self.policy_net.evaluate_actions(
            next_state_batch,
            action_batch,
        )
        # TODO: The tutorial reshapes the tensors. But we really don't need it I think
        self.learn(returns_batch, vals.squeeze(),
                   action_log_probs, dist_entropy)

    def learn(self, expected_vals, vals, action_log_prob, dist_entropy):
        self.broadcast(Event.BEGIN_LEARN)
        self.optimizer.zero_grad()
        # TODO: Use GAE
        adv = expected_vals - vals
        v_loss = adv.pow(2).mean()
        a_loss = -(adv.detach() * action_log_prob).mean()
        loss = v_loss * self.value_loss_coef + a_loss - dist_entropy * self.entropy_coef
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.broadcast(Event.END_LEARN)

    def to(self, torch_device):
        self.policy_net.to(torch_device)

    def save_net(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)

    def load_net(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath))

    def eval(self):
        self.policy_net.eval()

    def train(self):
        self.policy_net.train()

    def eval_episode(self, render=False, seed=None, load=''):
        rewards = deque()
        states = deque()

        env = DummyVecEnv([make_env(self.env_id, seed, 0)])
        self.set_global_seed(seed)

        if load is not '':
            self.load_net(load)

        done = False
        state = env.reset()
        while not done:
            if render:
                env.render()
            _, action, _ = self.act(state)
            obs, rew, done, _ = env.step(action)
            rewards.append(np.squeeze(rewards))
            states.append(np.reshape(state, -1))
            state = obs
        states = np.array(states)
        rewards = np.array(rewards).reshape(-1, 1)
        return np.append(states, rewards, axis=1)
