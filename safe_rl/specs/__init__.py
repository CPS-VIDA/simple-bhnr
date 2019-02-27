from .register import register, get_spec

from . import cartpole
from . import bipedal_walker

register('CartPole-v1', cartpole.SPEC, cartpole.SIGNALS)
register('BipedalWalker-v2', bipedal_walker.SPEC, bipedal_walker.SIGNALS)
