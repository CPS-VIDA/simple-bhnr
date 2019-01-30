"""Define basic memory"""

from abc import ABC, abstractmethod

from typing import NamedTuple
from collections import deque, namedtuple

import numpy as np

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))

QTransition = namedtuple(
    'QTransition', ('q',) + Transition._fields)


class BaseMemory(ABC):

    @abstractmethod
    def update(self, *args):
        pass

    @abstractmethod
    def push(self, *args):
        pass

    @abstractmethod
    def sample(self, size):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @property
    @abstractmethod
    def first(self):
        pass

    @property
    @abstractmethod
    def last(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    @abstractmethod
    def capacity(self):
        pass
