import random
from collections import deque, namedtuple

import numpy as np
from safe_rl.core.memory import BaseMemory, Transition, QTransition, ACTransition, AdvTransition
from safe_rl.memory.uniform import UniformReplay

import random


class RolloutMemory(UniformReplay):

    def sample(self, size=None):
        """
        Here, rather than uniform sampling, we get a slice of the memory of size.
        The order of the samples is the same as it was added into memory
        """
        samples = list(self.buffer)
        return samples[:size]


class MultiProcRolloutMemory(BaseMemory):

    def __init__(self, n_steps, n_procs):
        self.n_steps = n_steps
        self.n_procs = n_procs
        self.buffers = [deque(maxlen=n_steps) for _ in range(n_procs)]

    def update(self, *args):
        pass

    def push(self, items):
        assert isinstance(items, (tuple, list))
        assert len(items) == self.n_procs
        assert all(isinstance(item, ACTransition) for item in items)

        for i, buf in enumerate(self.buffers):
            buf.append(items[i])

    def sample(self, size):
        l = min([len(buf) for buf in self.buffers])
        tmp_buffer = [list(buf)[:l] for buf in self.buffers]
        tmp_buffer = [buf[:size] for buf in tmp_buffer]
        tmp_buffer = [ACTransition(*zip(*buf)) for buf in tmp_buffer]
        return tmp_buffer

    def __len__(self):
        return max([len(buf) for buf in self.buffers])

    def __iter__(self):
        return NotImplementedError

    def compute_returns(self, next_value, use_gae, gamma, tau):
        tmp_bf = [None] * self.n_procs  # type: list[deque]
        for i, buf in enumerate(self.buffers):
            next_val = next_value[i]
            tmp = tuple(zip(*buf))
            batch = ACTransition(*tmp)
            dones = np.array(batch.done, dtype=np.uint8)
            rew = np.array(batch.reward)
            ret = np.array(batch.returns)
            ret[-1] = next_val * gamma * (1 - dones[-1]) + rew[-1]
            for j in reversed(range(len(ret) - 2)):
                ret[j] = ret[j + 1] * gamma * (1 - dones[j]) + rew[j]
            new_batch = ACTransition(
                state=batch.state,
                action=batch.action,
                reward=batch.reward,
                next_state=batch.next_state,
                done=batch.done,
                action_log_prob=batch.action_log_prob,
                value_pred=batch.value_pred,
                returns=ret,
            )
            tmp_bf[i] = deque([ACTransition(*tr) for tr in zip(*new_batch)])
        self.buffers = tmp_bf

    def clear_until_last(self):
        last = self.last
        self.reset()
        self.push(last)

    @property
    def first(self):
        return [self.buffers[i][0] for i in range(self.n_procs)]

    @property
    def last(self):
        return [self.buffers[i][-1] for i in range(self.n_procs)]

    def reset(self):
        for buf in self.buffers:
            buf.clear()

    @property
    def capacity(self):
        return self.n_procs * self.n_steps

    def ppo_sample_generator(self, advantages, n_minibatch):
        batch_size = self.n_procs * self.n_steps
        assert batch_size >= n_minibatch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(self.n_procs, self.n_steps, batch_size, n_minibatch))

        minibatch_size = batch_size // n_minibatch
        batch = deque()
        for buf in self.buffers:
            batch.extend(buf)
        batch = ACTransition(*zip(*batch))
        batch_adv = AdvTransition(
            batch.state, batch.action, batch.reward, batch.next_state, batch.done, batch.action_log_prob,
            batch.value_pred, batch.returns, advantages
        )
        batch_adv = [AdvTransition(*tr) for tr in zip(*batch_adv)]
        for n in range(n_minibatch):
            yield AdvTransition(*zip(*random.sample(batch_adv, minibatch_size)))
