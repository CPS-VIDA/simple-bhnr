from .register import register, get_spec

from . import cartpole
from . import bipedal_walker
from .vrep import quadrotor_position_control

register('CartPole-v1', cartpole.SPEC, cartpole.SIGNALS)
register('BipedalWalker-v2', bipedal_walker.SPEC, bipedal_walker.SIGNALS)
register('QuadrotorPositionControlEnv-v0',
         quadrotor_position_control.SPEC, quadrotor_position_control.SIGNALS)
