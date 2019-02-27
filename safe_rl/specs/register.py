
SPEC_REGISTRY = dict()


def register(env_id, spec, signals):
    SPEC_REGISTRY[env_id] = (spec, signals)


def get_spec(env_id):
    if env_id in SPEC_REGISTRY:
        return SPEC_REGISTRY[env_id]
    raise ValueError('Given env id not found in registry: {}'.format(env_id))
