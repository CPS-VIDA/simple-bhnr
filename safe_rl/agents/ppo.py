import copy
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from safe_rl.core.agent import BaseAgent

from safe_rl.agents.a2c import A2C

from safe_rl.utils.general import make_env
from safe_rl.core.observers import Event
from safe_rl.envs import SubprocVecEnv, DummyVecEnv
from safe_rl.models.ac import ActorCritic
from safe_rl.core.memory import ACTransition
from safe_rl.memory.rollout import MultiProcRolloutMemory


class PPO(A2C):
    default_hyperparams = dict(A2C.default_hyperparams)
    default_hyperparams.update(dict(
        clipping=0.2,
        use_clipped_value_loss=True,
        ppo_epochs=4,
        n_minibatch=32,
    ))

    def __init__(self, env, hyperparams, **kwargs):
        super(PPO, self).__init__(env, hyperparams, **kwargs)

        self.clipping = self.hyp['clipping']
        self.ppo_epochs = self.hyp['ppo_epochs']
        self.n_minibatch = self.hyp['n_minibatch']
        self.use_clipped_value_loss = self.hyp['use_clipped_value_loss']

        assert self.n_minibatch <= self.n_workers * self.n_steps

        optim_fn = self.hyp.get('optim_fn',
                                lambda params: optim.Adam(params, lr=self.lr, eps=self.epsilon))
        self.optimizer = optim_fn(self.policy_net.parameters())

    def process_rollout(self):
        unroll_pre = ACTransition(*zip(*self.memory.sample(size=-1)))

        msr = np.array(unroll_pre.reward).mean(axis=0)
        sum_ret = np.array(unroll_pre.returns).mean(axis=0).sum()
        print('Mean Step Return: {}'.format(sum_ret))

        self.mean_step_rewards.extend(msr)
        self.episode_rewards += np.array(unroll_pre.reward).sum(axis=1)

        returns = torch.cat(tuple(torch.tensor(unroll_pre.returns))).squeeze()
        value_preds = torch.cat(tuple(torch.tensor(unroll_pre.value_pred))).squeeze()
        advantages = returns - value_preds
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        advantages = advantages.detach().cpu().numpy()

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epochs):
            data_gen = self.memory.ppo_sample_generator(advantages, self.n_minibatch)
            for sample in data_gen:
                obss = self._get_tensor(sample.next_state)
                acts = self._get_tensor(sample.action)
                val_p = self._get_tensor(sample.value_pred)
                alp = self._get_tensor(sample.action_log_prob)
                rets = self._get_tensor(sample.returns)
                adv = self._get_tensor(sample.advantage).view(-1, 1)
                dones = self._get_tensor(sample.done, torch.uint8)

                values, action_log_probs, dist_entropy = self.policy_net.evaluate_actions(obss, acts)

                ratio = torch.exp(action_log_probs - alp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clipping, 1.0 + self.clipping) * adv
                a_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = val_p + \
                                         (values - val_p).clamp(-self.clipping, self.clipping)
                    value_losses = (values - rets).pow(2)
                    value_losses_clipped = (value_pred_clipped - rets).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (rets - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + a_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += a_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epochs * self.n_minibatch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
