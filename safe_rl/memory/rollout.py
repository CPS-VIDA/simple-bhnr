import random
from collections import deque, namedtuple

import numpy as np
from safe_rl.core.memory import BaseMemory, Transition, QTransition
from safe_rl.memory.uniform import UniformReplay


class RolloutMemory(UniformReplay):

   def sample(self, size=None):
       """
       Here, rather than uniform sampling, we get a slice of the memory of size.
       The order of the samples is the same as it was added into memory
       """
       samples = list(self.buffer)
       return samples[:size]

