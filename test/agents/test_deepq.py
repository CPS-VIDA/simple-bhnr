import copy

import torch
import torch.nn as nn

from safe_rl.agents.deepq import DQN
from safe_rl.experiments.trial import trial_runner

AGENT_CONFIG = dict(
    env_id='CartPole-v1',
    name='default',
    n_trials=8,
    n_episodes=1000,
    n_eval_episodes=20,
    render=False,
    save_dir='data',
    backup_dir='backups',
    backup_interval=100,
    device=(torch.device('cuda')
            if torch.cuda.is_available() else torch.device('cpu')),
    net=nn.Sequential(
        nn.Linear(4, 24), nn.ReLU(),
        nn.Linear(24, 24), nn.ReLU(),
        nn.Linear(24, 2)
    ),
    hyperparams=dict(
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
    ),
    seed_fn = lambda cur_seed: 0 if cur_seed is None else cur_seed + 1
)


def gen_agent(conf):
    conf = conf.copy()
    env_id = conf.pop('env_id')
    net = copy.deepcopy(conf.pop('net'))
    hyp = conf.pop('hyperparams')
    return DQN(env_id, hyp, net, **conf)


def test_cartpole_deepq_uniform_notargetfix():
    hyp = dict()
    config = AGENT_CONFIG.copy()
    config['name'] = 'cartpole/vanilla/deepq_uniform_notargetfix'
    config['hyperparams'].update(hyp)
    trial_runner(gen_agent, config)


def test_cartpole_deepq_uniform_doubleq():
    hyp = dict(
        double_q=True,
    )
    config = AGENT_CONFIG.copy()
    config['name'] = 'cartpole/vanilla/deepq_uniform_doubleq'
    config['hyperparams'].update(hyp)
    trial_runner(gen_agent, config)


if __name__ == "__main__":
    # test_cartpole_deepq_uniform_doubleq()
    test_cartpole_deepq_uniform_notargetfix()
