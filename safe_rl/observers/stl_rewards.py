"""
In this file, we define an observer that modifies the temporary memory of the 
actor/agent that it attaches to in order to enable use of locally shaped reward 
functions using partial signal robustness
"""

from collections import deque

import numpy as np
from safe_rl.core.observers import BaseObserver, Event
from safe_rl.core.memory import BaseMemory, Transition, QTransition
from safe_rl.memory.rollout import RolloutMemory
from temporal_logic.signal_tl.semantics.base import BaseMonitor


class PartialSignal(BaseMemory):
    def __init__(self, max_len, monitor: BaseMonitor, agent_mem: BaseMemory):
        self.max_len = max_len
        self.monitor = monitor
        self.agent_mem = agent_mem
        self.buffer = deque(max_len)

        self._capacity = agent_mem.capacity


    def update(self, *args):
        self.agent_mem.update(args)

    def push(self, item):
        assert isinstance(item, Transition)
        self.buffer.append(item)
        is_done = item.done
        if (len(self.buffer) == self.max_len) or is_done:
            self._flush()
    
    def _flush(self):
        unzipped = Transition(*zip(*list(self.buffer)))
        partial_trace = np.array(unzipped.state)
        R = np.zeros(len(partial_trace))
        if len(R) > 2:
            R = self.monitor(partial_trace)
            # TODO: Check if we need to sum it up
        t_list = Transition(
            unzipped.state,
            unzipped.action,
            R,
            unzipped.next_state,
            unzipped.done,
        )
        for tr in zip(*t_list):
            self.agent_mem.push(Transition(*tr))
        self.buffer.clear()

    def sample(self, size):
        self.agent_mem.sample(size)

    def __len__(self):
        return len(self.agent_mem)

    def __iter__(self):
        return iter(self.agent_mem)

    @property
    def first(self):
        return self.agent_mem.first

    @property
    def last(self):
        return self.agent_mem.last

    def reset(self):
        self.buffer.clear()
        self.agent_mem.reset()

    @property
    def capacity(self):
        # TODO: Check this
        return self.agent_mem.capacity        
    

class STLRewarder(BaseObserver):
    def __init__(self, monitor, partial_sig_len):
        self.monitor = monitor
        self.partial_sig_len = partial_sig_len
        self.psig = None
        self.agent_mem = None
        self.step_counter = 0

    def attach(self, agent):
        if agent.rollout_len:
            self.partial_sig_len = min(self.partial_sig_len, agent.rollout_len)
        
        self.agent_mem = agent.memory
        self.psig = PartialSignal(self.partial_sig_len, self.monitor, self.agent_mem)
        agent.memory = self.psig
        self.agent = agent

    def notify(self, msg):
        if msg is Event.SYNC:
            # DONE: Actor Critic/Async handling
            # TODO: Check if this is enough
            self.psig._flush()
