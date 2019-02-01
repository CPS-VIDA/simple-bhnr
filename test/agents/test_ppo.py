import copy

import torch
import torch.nn as nn

from safe_rl.agents.ppo import PPO
from safe_rl.experiments.trial import trial_runner

AGENT_CONFIG = dict(
    env_id='MountainCarContinuous-v0',
    name='default',
    n_trials=8,
    n_episodes=10000,
    n_eval_episodes=20,
    render=False,
    save_dir='data',
    backup_dir='backups',
    backup_interval=100,
    device=(torch.device('cuda')
            if torch.cuda.is_available() else torch.device('cpu')),
    # net=ActorCritic(3, 1),
    hyperparams=dict(
        gamma=0.95,

        lr=0.001,
        epsilon=1e-5,
        alpha=0.99,

        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,

        n_steps=16,
        n_workers=4,

        use_gae=False,

        clipping=0.2,
        use_clipped_value_loss=True,
        ppo_epochs=4,
        n_minibatch=32,
    ),
    seed_fn=lambda cur_seed: 0 if cur_seed is None else cur_seed + 1
)


def gen_agent(conf):
    conf = conf.copy()
    env_id = conf.pop('env_id')
    # net = copy.deepcopy(conf.pop('net'))
    hyp = conf.pop('hyperparams')
    return PPO(env_id, hyp, **conf)


def test_pendulum_ppo():
    hyp = dict()
    config = AGENT_CONFIG.copy()
    config['name'] = 'pendulum/vanilla/ppo'
    config['hyperparams'].update(hyp)
    trial_runner(gen_agent, config)


if __name__ == "__main__":
    test_pendulum_ppo()