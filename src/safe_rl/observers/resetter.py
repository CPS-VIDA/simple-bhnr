from collections import deque

from safe_rl.core.observers import BaseObserver, Event


class ForceResetter(BaseObserver):

    def __init__(self, max_steps, n_procs=1):
        self.max_steps = max_steps
        self.n_procs = n_procs
        self.buffer = [deque(maxlen=max_steps)] * self.n_procs
        self.agent = None

    def attach(self, agent):
        n_procs = self.n_procs
        try:
            self.n_procs = min(agent.n_workers, n_procs)
        except:
            pass

        self.agent = agent

    def notify(self, msg):
        if msg is Event.STEP:
            last = self.agent.memory.last
            if isinstance(last, list) and hasattr(last[0], 'done'):
                for i, t in enumerate(last):
                    self.buffer[i].append(t.done)
            elif self.n_procs == 1 and hasattr(last, 'done'):
                self.buffer[0].append(last.done)

            reset = False
            for buf in self.buffer:
                if len(buf) >= self.max_steps and any(buf):  # If no dones in the last n steps
                    self.agent.env.reset()  # Force env reset
                    reset = True
                if reset:
                    buf.clear()

            if reset:
                print("RESETTING ENVS")
