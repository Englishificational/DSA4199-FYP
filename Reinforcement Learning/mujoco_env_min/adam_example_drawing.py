import gym
import argparse
import os
import gym
import numpy as np
import pickle
from tqdm import trange
import visdom
import math

import torch

from ddpg import DDPG
from normalized_actions import NormalizedActions
from action_noise import NormalActionNoise
from utils import save_model, vis_plot, load_model
import ast

import csv
import random
import time

method = "nr_mdp"
exploration_method = "nr_mdp"
beta = 0.9
learning_rate = 1e-3
gamma = 0.99
tau = 0.999
hidden_size_dim0 = 64
hidden_size_dim1 = 64
alpha = 0.999
replay_size = 1000000
optimizer = "Adam"

num_steps = 500000
num_epochs = None
num_epochs_cycles = 20 #20
num_rollout_steps = 100 #100
visualize = False
ratio = 0

batch_size = 128
number_of_train_steps = 50
two_player = True
Kt = 15

# added after looking at plot.ipynb
alpha = 0.1
method ="nr_mdp"
ratio = 1
epsilon = 1e-2 # i think only for SGLD?


# env_name = 'HalfCheetah-v4'
# env = gym.make(env_name)
# #saved_loc = os.getcwd() + '/my_test_archived/' + 'HalfCheetah-v4-arctan-c_magnitude-SGLMLD/SGLMLD_arctan_actor_c100/1' 
# saved_loc = os.getcwd() + '/my_test_archived/' + 'HalfCheetah-v4-arctan-c_magnitude-SGLMLD/SGLMLD_arctan_actor_c10000/1' 

# env_name = 'Walker2d-v4'
# env = gym.make(env_name)
# saved_loc = os.getcwd() + '/my_test_archived/' + 'Walker2d-v4-c_magnitude-SGLMLD/SGLD__thermal_0.01_action_noise_0.01/1' 
# #saved_loc = os.getcwd() + '/my_test_archived/' + 'Walker2d-v4-c_magnitude-SGLMLD/SGLMLD_actor_c100000/1' 

# env_name = "Hopper-v4"
# env = gym.make(env_name)
# #saved_loc = os.path.split(os.getcwd())[0] + '/mujoco_env_max/my_test_archived/Hopper-v4-squared-c_magnitude-SGLMLD/SGLD_thermal_0.001__action_noise_0.2/1'
# saved_loc = os.path.split(os.getcwd())[0] + '/mujoco_env_max/my_test_archived/Hopper-v4-squared-c_magnitude-SGLMLD/SGLMLD_squared_adversary_c10000/2'


env_name = "HalfCheetah-v4"
env = gym.make(env_name)
#saved_loc = os.path.split(os.getcwd())[0] + '/mujoco_env_specific/my_test_archived/HalfCheetah-v4-c_magnitude-SGLMLD/SGLMLD_ad_c1_act_c100000/1'
saved_loc = os.path.split(os.getcwd())[0] + '/mujoco_env_specific/my_test_archived/HalfCheetah-v4-c_magnitude-SGLMLD_2/SGLMLD_ad_c100000_act_c100000/1'

agent = DDPG(beta=beta, epsilon=epsilon, learning_rate=learning_rate, gamma=gamma, 
                tau=tau, hidden_size_dim0=hidden_size_dim0, hidden_size_dim1=hidden_size_dim1, 
                num_inputs=env.observation_space.shape[0], action_space=env.action_space, 
                train_mode=True, alpha=alpha, replay_size=replay_size, optimizer=optimizer, 
                two_player=True)


load_model(agent, saved_loc)

env.reset()
env.seed(42)
env.render()
time.sleep(2)
state = agent.Tensor([env.reset()])  
for i in range(600):
  if i % 1000 == 0:
    print(i)
  env.render()

  
  action = agent.select_action(state)#, action_noise=normalnoise, mdp_type=exploration_method)
  state, reward, done, _ = env.step(action.cpu().numpy()[0])
  state = agent.Tensor([state])
  if random.random() < 0.005: 
    print("Reward:", reward)
  #agent.actor
env.close()