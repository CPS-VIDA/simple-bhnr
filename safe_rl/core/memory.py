"""Define basic memory"""

from abc import ABC, abstractmethod
from collections import deque, namedtuple
from typing import NamedTuple

import numpy as np

Transition = namedtuple(
    'Transition', (
        'state', 'action', 'reward', 'next_state', 'done'
    )
)

QTransition = namedtuple(
    'QTransition', (
        'state', 'action', 'reward', 'next_state', 'done',
        'q',
    )
)

ACTransition = namedtuple(
    'ACTransition', (
        'state', 'action', 'reward', 'next_state', 'done',
        'action_log_prob', 'value_pred', 'returns'
    )
)

AdvTransition = namedtuple(
    'AdvTransition', (
        'state', 'action', 'reward', 'next_state', 'done',
        'action_log_prob', 'value_pred', 'returns',
        'advantage'
    )
)


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
