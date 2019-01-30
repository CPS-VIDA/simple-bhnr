"""Defines an abstract observer.

An observer is an class that attaches to an actor/agent modifies the information in the actor/agent when notified with
appropriate messages.
"""
from abc import ABC, abstractmethod
from enum import Enum, auto


class Event(Enum):
    BEGIN_TRAINING = auto()
    BEGIN_EPISODE = auto()
    STEP = auto()
    BEGIN_LEARN = auto()
    SYNC = auto()
    END_LEARN = auto()
    END_EPISODE = auto()
    END_TRAINING = auto()


class BaseObserver(ABC):
    agent = None

    @abstractmethod
    def attach(self, agent_or_actor):
        pass

    @abstractmethod
    def notify(self, msg):
        pass
