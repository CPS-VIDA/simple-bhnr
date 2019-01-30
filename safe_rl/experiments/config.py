import torch


class Config:
    name = 'default'

    @staticmethod
    def agent_fn(env, ): return None

    @staticmethod
    def env_fn(): return None

    @staticmethod
    def eval_env_fn(): return None

    n_trials = 1
    n_episodes = 1000
    n_eval_episodes = 10
    n_workers = 1

    render = False

    loss_fn = None

    @staticmethod
    def optim_fn(params): return None

    @staticmethod
    def actor_optim_fn(params): return None

    @staticmethod
    def critic_optim_fn(params): return None

    @staticmethod
    def net_fn(): return None

    device = (torch.device('cuda')
              if torch.cuda.is_available() else torch.device('cpu'))

    logger = None
    save_dir = None
    backup_dir = None
    backup_interval = 0
    log_dir = None
