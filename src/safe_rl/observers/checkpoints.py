import os
from collections import deque
from datetime import datetime

import numpy as np

from safe_rl.core.observers import BaseObserver, Event


class EpisodicCheckpointSaver(BaseObserver):
    def __init__(self, dir_path, prefix='checkpoint', interval=100, max_num_files=10):
        self.dir_path = os.path.abspath(dir_path)
        os.makedirs(self.dir_path, exist_ok=True)

        self.prefix = prefix
        self.interval = interval
        self.max_num_files = max_num_files
        self.episode_count = 0
        self.file_count = 0

        self.agent = None

    def notify(self, msg):
        if msg is Event.END_EPISODE:
            self.episode_count += 1
            if self.episode_count % self.interval == 0:
                if self.max_num_files >= self.file_count:
                    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                    filename = '{}.{}.{}.pt'.format(
                        now, self.prefix, self.episode_count)
                    pth = os.path.join(self.dir_path, filename)
                    self.agent.save_net(pth)
                    self.file_count += 1
                filename = '{}.latest.pt'.format(self.prefix)
                pth = os.path.join(self.dir_path, filename)
                self.agent.save_net(pth)

    def attach(self, agent):
        self.agent = agent

