"""Define basic memory"""

from collections import deque


from abc import ABC, abstractmethod


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
    def size(self):
        pass

    @property
    @abstractmethod
    def capacity(self):
        pass
