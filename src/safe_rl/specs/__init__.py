from .register import register, get_spec

from temporal_logic.signal_tl.semantics import FilteringMonitor, EfficientRobustnessMonitor

from drone_gym.envs import VREPQuadrotorPositionControlEnv, SimpleQuadrotorPositionControlEnv

from . import cartpole
from . import bipedal_walker
from .vrep import quadrotor_position_control


register('CartPole-v1', cartpole.SPEC, cartpole.SIGNALS, FilteringMonitor)


register('BipedalWalker-v2',
         bipedal_walker.SPEC,
         bipedal_walker.SIGNALS,
         FilteringMonitor)


register('VREPQuadrotorPositionControlEnv-v0',
         quadrotor_position_control.SPEC,
         quadrotor_position_control.SIGNALS,
         EfficientRobustnessMonitor)

register('SimpleQuadrotorPositionControlEnv-v0',
         quadrotor_position_control.SPEC,
         quadrotor_position_control.SIGNALS,
         EfficientRobustnessMonitor)

