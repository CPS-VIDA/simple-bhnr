from abc import ABC, abstractmethod
import os

import numpy as np


class BaseAgent(ABC):
    """Abstract Agent class"""

    @abstractmethod
    def act_and_train(self, obs, reward):
        """
        Select an action for training

        Returns:
            ~object: action
        """
        pass

