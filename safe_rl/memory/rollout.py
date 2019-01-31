import random
from collections import deque, namedtuple

import numpy as np
from safe_rl.core.memory import BaseMemory, Transition, QTransition, ACTransition
from safe_rl.memory.uniform import UniformReplay


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
        tmp_buffer = [ACTransition(*zip(*buf)) for buf in self.buffers]
        return tmp_buffer

    def __len__(self):
        return max([len(buf) for buf in self.buffers])

    def __iter__(self):
        return NotImplementedError

    def compute_returns(self, next_value, use_gae, gamma, tau):
        tmp_bf = [None] * self.n_procs  # type: list[deque]
        for i, buf in enumerate(self.buffers):
            next_val = next_value[i]
            batch = ACTransition(*zip(*buf))
            dones = np.array(batch.done,dtype=np.uint8)
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
