import copy
import os
from datetime import datetime
import argparse

import pandas as pd
import torch
import torch.nn as nn

from safe_rl.agents.ppo import PPO
import temporal_logic.signal_tl as stl
from temporal_logic.signal_tl.semantics import FilteringMonitor, EfficientRobustnessMonitor

from safe_rl.agents.ppo import PPO
from safe_rl.experiments.trial import trial_runner
from safe_rl.observers.multi_proc_stl import MultiProcSTLRewarder
from safe_rl.observers.checkpoints import EpisodicCheckpointSaver
from safe_rl.utils.general import set_global_seed

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
        (vel_x > 1),
        # Reduce hull tilt to ~ +- 12deg
        (abs(hull_angle) <= 0.2),
        # Reduce hull angular velocity
        (abs(hull_angularVelocity) < 2),
        # Prevents running
        stl.Implies((leg_1_ground_contact_flag > 0),
                    stl.And(
                        stl.F(leg_2_ground_contact_flag > 0, (10, 16)),  # Set foot in 10-16 timesteps
                        stl.G(leg_2_ground_contact_flag <= 0, (1, 8)),  # Make sure to actually lift leg
                    )),
        stl.Implies((leg_2_ground_contact_flag > 0),
                    stl.And(
                        stl.F(leg_1_ground_contact_flag > 0, (10, 16)),  # Set foot in 10-16 timesteps
                        stl.G(leg_1_ground_contact_flag <= 0, (1, 8)),  # Make sure to actually lift leg
                    )),
    )
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run PPO')
    parser.add_argument('--env-id', required=True, help='Gym env id')
    parser.add_argument('--name', required=True, help='Name of the current experiment')
    # parser.add_argument('--n-trials', default=8, type=int, help='Number of tr')
    parser.add_argument('--maximum-steps', default=100000, type=int, help='Total number of steps desired')
    parser.add_argument('--n-eval-episodes', default=100, type=int, help='Number of episodes to evaluate the policy')
    parser.add_argument('--render', action='store_true', default=False, help='Render the env?')
    parser.add_argument('--save-dir', required=True, help='Base directory for all your data')
    parser.add_argument('--backup-dir', required=True, help='Base dir for all backup data')
    parser.add_argument('--backup-interval', default=100, type=int)
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--gamma', type=float, default=0.95)
    
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.99)
    
    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    
    parser.add_argument('--n-steps', type=int, default=32)
    parser.add_argument('--n-workers', type=int, default=4)
    
    parser.add_argument('--use-gae', action='store_true', default=False)
    
    parser.add_argument('--clipping', type=float, default=0.2)
    parser.add_argument('--use-clipped', action='store_true', default=True)
    parser.add_argument('--ppo-epoch', type=int, default=4)
    parser.add_argument('--n-minibatch', type=int, default=32)
    
    parser.add_argument('--seed', default=None)

    parser.add_argument('--use-stl', action='store_true', default=False)

    args = parser.parse_args()
    config = dict(
        env_id=args.env_id,
        name=args.name,
        maximum_steps=args.maximum_steps,
        n_eval_episodes=args.n_eval_episodes,
        render=args.render,
        save_dir=args.save_dir,
        backup_dir=args.backup_dir,
        backup_interval=args.backup_interval,
        device=torch.device(args.device),
        hyperparams=dict(
            gamma=args.gamma,

            lr=args.lr,
            epsilon=args.epsilon,
            alpha=args.alpha,

            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,

            n_steps=args.n_steps,
            n_workers=args.n_workers,

            use_gae=args.use_gae,

            clipping=args.clipping,
            use_clipped_value_loss=args.use_clipped,
            ppo_epochs=args.ppo_epoch,
            n_minibatch=args.n_minibatch,
        ),
        seed=args.seed,
        use_stl=args.use_stl,
    )
    return config


def gen_agent(conf):
    conf = conf.copy()
    env_id = conf.pop('env_id')
    # net = copy.deepcopy(conf.pop('net'))
    hyp = conf.pop('hyperparams')
    agent = PPO(env_id, hyp, **conf)
    if conf['use_stl']:
        monitor = FilteringMonitor(SPEC, SIGNALS)
        n_steps = conf['hyperparams']['n_steps']
        n_workers = conf['hyperparams']['n_workers']
        agent.attach(MultiProcSTLRewarder(n_steps, n_workers, monitor))
    return agent


def run_training(conf):
    env_id = conf['env_id']
    name = conf['name']
    maximum_steps = conf['maximum_steps']
    n_eval_episodes = conf['n_eval_episodes']
    render = conf['render']
    save_dir = conf['save_dir']
    backup_dir = conf['backup_dir']
    backup_interval = conf['backup_interval']
    device = conf['device']
    seed = conf['seed']

    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    trial_dir = os.path.join(save_dir, name)
    check_dir = os.path.join(backup_dir, name, now)
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)

    agent = gen_agent(conf)
    agent.attach(EpisodicCheckpointSaver(check_dir, interval=backup_interval))
    set_global_seed(seed)

    data = agent.run_training(maximum_steps, render=render)
    if data is not None:
        df = pd.DataFrame(data)
        now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        mon_csv = os.path.join(trial_dir, now + '.monitor.csv')
        df.to_csv(mon_csv, header=False)


if __name__ == "__main__":
    run_training(parse_args())
