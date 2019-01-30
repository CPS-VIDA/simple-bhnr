from safe_rl.core.observers import BaseObserver, Event
import numpy as np


class EpsilonGreedyUpdater(BaseObserver):

    def __init__(self, epsilon_min, epsilon_decay):
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.agent = None

    def attach(self, agent):
        assert agent.epsilon is not None, "Epsilon attribute doesn't exist in this agent"
        self.agent = agent

    def notify(self, msg):
        if msg is Event.END_EPISODE:
            self.agent.epsilon = max(self.epsilon_min, self.agent.epsilon * self.epsilon_decay)


class TargetUpdater(BaseObserver):

    def __init__(self, target_update):
        self.step_count = 0
        self.target_update = target_update

    def attach(self, agent):
        assert agent.policy_net is not None, "Policy net hasn't been initialized"
        assert agent.target_net is not None, "Target net hasn't been initialized"
        self.agent = agent

    def notify(self, msg):
        if msg is Event.STEP:
            self.step_count += 1
            if self.step_count % self.target_update == 0:
                self.agent.target_net.load_state_dict(
                    self.agent.policy_net.state_dict())
