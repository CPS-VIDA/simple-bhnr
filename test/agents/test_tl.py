import copy

import temporal_logic.signal_tl as stl
import torch
import torch.nn as nn
from safe_rl.agents.deepq import DQN
from safe_rl.observers.stl_rewards import STLRewarder
from temporal_logic.signal_tl.semantics import FilteringMonitor

HYPERPARAMS = dict(
    gamma=0.95,

    lr=0.001,

    batch_size=32,

    double_q=False,
    target_net=False,
    target_update=15,

    memory_size=2000,
    memory_type='uniform',  # or per

    epsilon_greedy=True,
    epsilon=1.0,
    epsilon_min=0.1,
    epsilon_decay=0.95,
)

NET = nn.Sequential(
    nn.Linear(4, 24), nn.ReLU(),
    nn.Linear(24, 24), nn.ReLU(),
    nn.Linear(24, 2)
)

SIGNALS = ('x', 'x_dot', 'theta', 'theta_dot')
x, x_dot, theta, theta_dot = stl.signals(SIGNALS)
SPEC = stl.G(stl.F(x_dot < abs(0.01)) & (abs(theta) < 5) & (abs(x) < 0.5))

N_EPISODES = 200


# def test_cartpole_deepq_uniform_notargetfix_stl():
#     hyp = {}
#     net = copy.deepcopy(NET)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     device = torch.device(device)
#     agent = DQN('CartPole-v1', hyp, net, device=device)
#     agent.train(N_EPISODES, render=True)


def test_cartpole_deepq_uniform_doubleq_stl():
    hyp = dict(
        double_q=True,
    )
    net = copy.deepcopy(NET)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    agent = DQN('CartPole-v1', hyp, net, device=device)

    monitor = FilteringMonitor(SPEC, SIGNALS)
    agent.attach(STLRewarder(monitor, 15))

    agent.train(N_EPISODES, render=True)


if __name__ == "__main__":
    test_cartpole_deepq_uniform_doubleq_stl()
