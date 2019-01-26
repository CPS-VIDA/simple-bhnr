"""Defines an abstract observer.

An observer is an class that attaches to an actor/agent modifies the information in the actor/agent when notified with
appropriate messages.
"""
from abc import ABC, abstractmethod
from enum import Enum, auto


class Msg(Enum):
    BEGIN_TRAINING = auto()
    BEGIN_EPISODE = auto()
    STEP = auto()
    LEARN = auto()
    END_EPISODE = auto()
    END_TRAINING = auto()


class BaseObserver(ABC):

    @abstractmethod
    def attach(self, agent_or_actor):
        pass

    @abstractmethod
    def notify(self, msg):
        pass
