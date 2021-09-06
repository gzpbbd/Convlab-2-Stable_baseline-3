import os
import gym
from gym import spaces
import numpy as np
import logging


from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.dialog_agent import PipelineAgent
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.dialog_agent.env import Environment
from convlab2.policy.vector.vector_multiwoz import MultiWozVector

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

os.environ["CUDA_VISIBLE_DEVICES"] = ''


class GymEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['numpy']}

    def __init__(self, dataset_vector: MultiWozVector, environment: Environment, max_len=50):
        super(GymEnvironment, self).__init__()
        self.action_space = spaces.MultiBinary(dataset_vector.da_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dataset_vector.state_dim,),
                                            dtype=np.float64)

        self.dataset_vector = dataset_vector
        self.environment = environment
        self.current_state = dict()
        self.max_len = max_len
        self.counter = 0

    def step(self, action: np.ndarray):
        dict_action = self.dataset_vector.action_devectorize(action)
        state, reward, terminated = self.environment.step(dict_action)
        vec_state = self.dataset_vector.state_vectorize(state)
        info = dict()

        self.current_state = state
        self.counter += 1
        if self.counter >= self.max_len:
            terminated = True
            self.counter = 0
        return vec_state, reward, terminated, info

    def reset(self):
        state = self.environment.reset()
        vec_state = self.dataset_vector.state_vectorize(state)

        self.current_state = state
        self.counter = 0
        return vec_state

    def render(self, mode='human'):
        return self.current_state

    def close(self):
        self.current_state = dict()




if __name__ == '__main__':
    # get vector for dataset
    root_dir = os.path.dirname(os.path.abspath(__file__))
    voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
    voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
    vector = MultiWozVector(voc_file, voc_opp_file)

    # get environment
    # We don't need NLU, DST and NLG for user simulator
    policy_usr = RulePolicy(character='usr')
    simulator = PipelineAgent(None, None, policy_usr, None, 'user')
    # We don't need NLU and NLG for system
    dst_sys = RuleDST()
    evaluator = MultiWozEvaluator()
    conv_env = Environment(None, simulator, None, dst_sys, evaluator)

    env = GymEnvironment(vector, conv_env)

    logger = logging.getLogger()
    logger.setLevel('ERROR')
    ppo = PPO('MlpPolicy', env, verbose=1, n_steps=64, batch_size=64)
    ppo.learn(total_timesteps=10000)
