import os

os.environ["CUDA_VISIBLE_DEVICES"] = ''
from utils import calculate_time

import random
import numpy as np
import torch
import logging

from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.ppo import PPO
from convlab2.dialog_agent import PipelineAgent
from convlab2.util.analysis_tool.analyzer import Analyzer
from convlab2.util.train_util import init_logging_handler


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


@calculate_time
def test_end2end():
    # go to README.md of each model for more information
    sys_dst = RuleDST()
    sys_policy = PPO()
    # todo
    # sys_policy.load('/home/huangchenping/github_repo/ConvLab-2-master/convlab2/policy/mle/multiwoz/save/best_mle')

    sys_policy.load('/home/huangchenping/github_repo/ConvLab-2-master/convlab2/policy/ppo/save/{}'.format(filename))
    sys_agent = PipelineAgent(None, sys_dst, sys_policy, None, name='sys')

    user_dst = None
    user_policy = RulePolicy(character='usr')
    user_agent = PipelineAgent(None, user_dst, user_policy, None, name='user')

    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    set_seed(20200202)
    # todo
    analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name='None-RuleDST-{}-None'.format(filename),
                                   total_dialog=1000)
    # analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name='None-RuleDST-PPOPolicy-None', total_dialog=1000)


if __name__ == '__main__':
    # todo
    filename = '104_ppo'
    init_logging_handler('log', '_test_my_{}'.format(filename))
    test_end2end()
