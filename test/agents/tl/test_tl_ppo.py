"""Test STL+PPO"""
import copy

import mpmath
import temporal_logic.signal_tl as stl
import torch
import torch.nn as nn
from temporal_logic.signal_tl.semantics import FilteringMonitor, EfficientRobustnessMonitor

from safe_rl.agents.ppo import PPO
from safe_rl.experiments.trial import trial_runner
from safe_rl.observers.multi_proc_stl import MultiProcSTLRewarder

"""
BipedalWalker environment

Type: Box(24)

Num   | Observation                |  Min   |   Max  | Mean
------|----------------------------|--------|--------|------   
0     | hull_angle                 |  0     |  2*pi  |  0.5
1     | hull_angularVelocity       |  -inf  |  +inf  |  -
2     | vel_x                      |  -1    |  +1    |  -
3     |  vel_y                     |  -1    |  +1    |  -
4     | hip_joint_1_angle          |  -inf  |  +inf  |  -
5     | hip_joint_1_speed          |  -inf  |  +inf  |  -
6     | knee_joint_1_angle         |  -inf  |  +inf  |  -
7     | knee_joint_1_speed         |  -inf  |  +inf  |  -
8     | leg_1_ground_contact_flag  |  0     |  1     |  -
9     | hip_joint_2_angle          |  -inf  |  +inf  |  -
10    | hip_joint_2_speed          |  -inf  |  +inf  |  -
11    | knee_joint_2_angle         |  -inf  |  +inf  |  -
12    | knee_joint_2_speed         |  -inf  |  +inf  |  -
13    | leg_2_ground_contact_flag  |  0     |  1     |  -
14-23 | 10 lidar readings          |  -inf  |  +inf  |  -
"""

SIGNALS = (
    'hull_angle',
    'hull_angularVelocity',
    'vel_x',
    'vel_y',
    'hip_joint_1_angle',
    'hip_joint_1_speed',
    'knee_joint_1_angle',
    'knee_joint_1_speed',
    'leg_1_ground_contact_flag',
    'hip_joint_2_angle',
    'hip_joint_2_speed',
    'knee_joint_2_angle',
    'knee_joint_2_speed',
    'leg_2_ground_contact_flag',
    'lidar0',
    'lidar1',
    'lidar2',
    'lidar3',
    'lidar4',
    'lidar5',
    'lidar6',
    'lidar7',
    'lidar8',
    'lidar9',
)

(hull_angle, hull_angularVelocity, vel_x, vel_y, hip_joint_1_angle, hip_joint_1_speed, knee_joint_1_angle,
 knee_joint_1_speed, leg_1_ground_contact_flag, hip_joint_2_angle, hip_joint_2_speed, knee_joint_2_angle,
 knee_joint_2_speed, leg_2_ground_contact_flag, lidar0, lidar1, lidar2, lidar3, lidar4, lidar5, lidar6, lidar7, lidar8,
 lidar9) = stl.signals(
    SIGNALS)

SPEC = stl.G(
    stl.And(
        # Be moving forward always
        (vel_x > 0),
        # Reduce hull tilt to ~ +- 12deg
        (abs(hull_angle) <= 0.2),
        # Reduce hull angular velocity
        (abs(hull_angularVelocity) < 2),
        # Prevents running
        stl.Implies((leg_1_ground_contact_flag > 0),
                    stl.F(leg_2_ground_contact_flag <= 0, (0, 16))),
        stl.Implies((leg_2_ground_contact_flag > 0),
                    stl.F(leg_1_ground_contact_flag <= 0, (0, 16))),
    )
)

AGENT_CONFIG = dict(
    env_id='BipedalWalker-v2',
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
    hyp = conf.pop('hyperparams')
    agent = PPO(env_id, hyp, **conf)

    monitor = FilteringMonitor(SPEC, SIGNALS)
    n_steps = AGENT_CONFIG['hyperparams']['n_steps']
    n_workers = AGENT_CONFIG['hyperparams']['n_workers']
    agent.attach(MultiProcSTLRewarder(n_steps, n_workers, monitor))
    return agent


def test_bipedalwalker_ppo_stl():
    hyp = dict()
    config = AGENT_CONFIG.copy()
    config['name'] = 'pendulum/stl/ppo'
    config['hyperparams'].update(hyp)
    trial_runner(gen_agent, config)


if __name__ == "__main__":
    test_bipedalwalker_ppo_stl()
