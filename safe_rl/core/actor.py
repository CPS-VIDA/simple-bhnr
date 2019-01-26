import torch
import torch.multiprocessing as mp
import copy

import gym

from abc import abstractmethod, ABC


from enum import Enum, auto



class BaseActor(ABC, mp.Process):
    default_hyperparams = dict()

    def __init__(self, env, hyperparams):
        self.env = env
        self.hyp = self.default_hyperparams
        self.hyp.update(hyperparams)

    @abstractmethod
    def run(self, conn):
        pass
