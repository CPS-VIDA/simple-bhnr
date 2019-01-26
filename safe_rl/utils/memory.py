import random
from collections import namedtuple, deque

import numpy as np

from safe_rl.core.memory import BaseMemory
from .sumtree import SumTree

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

QTransition = namedtuple('QTransition', ['q', 'state', 'action', 'reward', 'next_state', 'done'])


class UniformReplay(BaseMemory):
    """A basic memory class"""

    def __init__(self, capacity):
        self._capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, item):
        self.buffer.append(item)

    def sample(self, size):
        return random.sample(self.buffer, size)

    def update(self, agent):
        pass

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    @property
    def first(self):
        return self.buffer[0]

    @property
    def last(self):
        return self.buffer[-1]

    @property
    def capacity(self):
        return self._capacity

    @capacity.setter
    def capacity(self, new_cap):
        self._capacity = new_cap
        self.buffer = deque(self.buffer, maxlen=self.capacity)

    def reset(self):
        self.buffer.clear()

    @property
    def size(self):
        return len(self.buffer)


class PrioritizedMemory(BaseMemory):
    """
    Prioritized memory that uses a SumTree for replay
    """
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self._capacity = capacity
        self.buffer = SumTree(self.capacity)
        self._p_max = 1
        self._last_idx = []
        self._last_is_w = []

    def _reset(self):
        self.buffer = SumTree(self.capacity)
        self._p_max = 1
        self._last_idx = []
        self._last_is_w = []

    @property
    def last_idx_weights(self):
        return self._last_idx, self._last_is_w

    @property
    def capacity(self):
        return self._capacity

    @property
    def max_priority(self):
        return self._p_max

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def update(self, idx, err):
        p = self._get_priority(err)
        self._p_max = max(p, self.max_priority)
        self.buffer.update(idx, p)

    def push(self, item, error=None):
        if error is None:
            error = self.max_priority
        priority = self._get_priority(error)
        self._p_max = max(priority, self.max_priority)
        self.buffer.add(item, priority)

    def sample(self, size):
        batch = [None] * size
        idxs = np.zeros(size, dtype=np.uint)
        priorities = np.zeros(size)
        seg_size = self.buffer.total / size

        for i in range(size):
            a = seg_size * i
            b = seg_size * (i + 1)
            num = np.random.uniform(a, b)
            idx, p, data = self.buffer.sample(num)
            batch[i] = data
            idxs[i] = int(idx)
            priorities[i] = p

        sampling_p = priorities / self.buffer.total
        is_weights = np.power(len(self.buffer) * sampling_p, -self.beta)
        is_weights /= is_weights.max()

        self._last_idx = idxs
        self._last_is_w = is_weights

        return batch

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def first(self):
        pass

    def last(self):
        pass

    def reset(self):
        self._reset()

    @property
    def size(self):
        return len(self.buffer)
