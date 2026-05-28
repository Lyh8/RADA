import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s: %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger
        self.use_tb = False
        self.use_sacred = False
        self.stats = {}

    def setup_tb(self, directory_name):
        self.writer = SummaryWriter(log_dir=directory_name)
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key] = (value, t)
        if self.use_tb: self.writer.add_scalar(key, value, t)
        if self.use_sacred and to_sacred:
            if key in self.sacred_info: self.sacred_info[key].append((t, value))
            else: self.sacred_info[key] = [(t, value)]

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>8} | Episode: {:>8}".format(*self.stats["episode"])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode": continue
            i += 1
            window = 5 if k != "epsilon" else 1
            if self.use_sacred and k in self.sacred_info:
                item = "{:.4f}".format(np.mean([x[1] for x in self.sacred_info[k][-window:]]))
            else: item = "{:.4f}".format(v[0])
            log_str += " | {}: {}".format(k, item)
        self.console_logger.info(log_str)