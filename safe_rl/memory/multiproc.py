import random
from collections import deque, namedtuple

import numpy as np
from safe_rl.core.memory import BaseMemory, Transition, QTransition


class MultiProcRolloutMemory(BaseMemory):

    def __init__(self, n_procs, capacity=None):
        self.n_procs = n_procs
        self._capacity = capacity if capacity is not None else np.inf
        self.buffers = [deque(maxlen=capacity)]*n_procs

    def update(self, *args):
        pass

    def push(self, transition_list):
        pass

    def sample(self, size):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass

    @property
    def first(self):
        pass

    @property
    def last(self):
        pass

    def reset(self):
        pass

    @property
    def capacity(self):
        return self._capacity

    
