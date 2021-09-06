import os
import time
import logging
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_logging_handler(log_dir, extra=''):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    format_str = '%(levelname)s\t %(asctime)s:  %(message)s\n                                                                      - %(filename)s (%(funcName)s %(lineno)d)'
    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(logging.Formatter(format_str,
                                                  datefmt='%H:%M:%S'))
    file_handler = logging.FileHandler('{}/log_{}.txt'.format(log_dir, current_time + extra))
    file_handler.setFormatter(logging.Formatter(format_str,
                                                datefmt='%H:%M:%S'))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logging.info('saving log to {}'.format(file_handler.baseFilename))


def to_device(data):
    if type(data) == dict:
        for k, v in data.items():
            data[k] = v.to(device=DEVICE)
    else:
        for idx, item in enumerate(data):
            data[idx] = item.to(device=DEVICE)
    return data
