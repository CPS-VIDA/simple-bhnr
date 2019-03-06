import random
from collections import deque, namedtuple

import numpy as np
from safe_rl.core.memory import BaseMemory, Transition, QTransition


class UniformReplay(BaseMemory):
    """A basic memory class"""

    def __init__(self, capacity):
        self._capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, item):
        self.buffer.append(item)

    def sample(self, size):
        return random.sample(self.buffer, size)

    def update(self, *args):
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

