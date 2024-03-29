from collections import deque

import numpy as np
from safe_rl.core.observers import BaseObserver, Event
from safe_rl.core.memory import BaseMemory, Transition, QTransition, ACTransition, AdvTransition
from safe_rl.memory.rollout import RolloutMemory, MultiProcRolloutMemory
from temporal_logic.signal_tl.semantics.base import BaseMonitor


class MultiProcPartialSignal(MultiProcRolloutMemory):
    def __init__(self, n_steps, n_procs, monitor: BaseMonitor):
        super(MultiProcPartialSignal, self).__init__(n_steps, n_procs)
        self.monitor = monitor
        self.psigs = [deque(maxlen=n_steps) for _ in range(self.n_procs)]

    def push(self, items):
        assert isinstance(items, (tuple, list))
        assert len(items) == self.n_procs
        assert all(isinstance(item, ACTransition) for item in items)

        for i, ps in enumerate(self.psigs):
            tr = items[i]
            ps.append(tr)
            if tr.done:
                self.flush(i)
                ps.clear()

    def flush(self, idx):
        sig_to_flush = list(self.psigs[idx])
        if len(sig_to_flush) == 0:
            return 
        unzipped = ACTransition(*zip(*sig_to_flush))
        partial_trace = np.array(unzipped.state)
        R = np.zeros(len(partial_trace))
        if len(R) > 2:
            R = self.monitor(partial_trace)
        t_list = ACTransition(
            unzipped.state,
            unzipped.action,
            R,
            unzipped.next_state,
            unzipped.done,
            unzipped.action_log_prob,
            unzipped.value_pred,
            unzipped.returns
        )
        for tr in zip(*t_list):
            self.buffers[idx].append(tr)
        self.psigs[idx].clear()

    @property
    def first(self):
        if len(self.psigs[0]) > 0:
            return [psig[0] for psig in self.psigs]
        return [self.buffers[i][0] for i in range(self.n_procs)]

    @property
    def last(self):
        if len(self.psigs[0]) > 0:
            return [psig[-1] for psig in self.psigs]
        return [buf[-1] for buf in self.buffers]


class MultiProcSTLRewarder(BaseObserver):
    def __init__(self, n_steps, n_workers, monitor):
        self.monitor = monitor
        self.n_steps = n_steps
        self.n_workers = n_workers
        self.psigs = None

    def attach(self, agent):
        assert isinstance(agent.memory, MultiProcRolloutMemory)

        self.n_steps = min(self.n_steps, agent.memory.n_steps)

        self.agent_mem = agent.memory  # type: MultiProcRolloutMemory
        self.psigs = MultiProcPartialSignal(self.n_steps, self.n_workers, self.monitor)
        agent.memory = self.psigs
        self.agent = agent

    def notify(self, msg):
        if msg is Event.SYNC:
            for i in range(self.n_workers):
                self.psigs.flush(i)

