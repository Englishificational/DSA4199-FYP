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
from utils import save_model, vis_plot
import ast

import itertools
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd

#env = NormalizedActions(gym.make("Walker2d-v4"))

# testing load pickle
results = {}
exp1 = "RMSProp\\no_noise"
if True:
    results[exp1] = {}
    cwd = os.getcwd()
    cwd += '\\my_test'+ '\\22' + '\\results'
    print(cwd)

    def load_dict(filename_to_load):
        with open(filename_to_load, 'rb') as f:
            test = pickle.load(f)
        return test

    results[exp1][0] = load_dict(cwd)

#print(results)

if 5:
    print("yes")
else:
    print("no")

# need to copy
Walker_exp = ['RMSprop/no_noise', 'ExtraAdam/action_noise_0.3', 'SGLD_thermal_0.01/action_noise_0.01']
HalfCheetah_exp = ['RMSprop/action_noise_0.2', 'ExtraAdam/action_noise_0.01', 'SGLD_thermal_0.01/no_noise']
exp = [Walker_exp, HalfCheetah_exp]
# in 
def plot_learning_all():
    return
def plot_learning_average():
    return
#plot_learning_all(exp, 'Comparison', comparison = True, OnePlayer = False, best = False)

envs = ['Walker2d-v4', 'HalfCheetah-v2', 'Hopper-v2']
for i, env in enumerate(envs):
    print(i)
    print(env)
#plot_learning_average(env, ax, exp, window_size = 3000, var = True, OnePlayer = False)
env = envs[0]
exp = Walker_exp

seed = 42
base_dir = os.getcwd()
base_dir += '\\models\\' + 'Walker2d-v4'
base_dir += '\\' + 'RMSprop\\no_noise' + '\\nr_mdp_0.1_1\\' + str(seed) + '\\results'
print(base_dir)
target = "mujoco_env\models\Walker2d-v4\RMSprop\no_noise\nr_mdp_0.1_1\42\results"

#Walker_exp params
alpha = 0.1
optimizer = "RMSProp"
action_noise = None
method ="nr_mdp"
ratio = 1

#def plot_learning_curves(results, ax, env_name, window_size, var, OnePlayer, best):    
window_size = 3000
colors = ['#396ab1', '#3e9651', '#cc2529', '#396ab1', '#da7c30', '#94823d', '#535154', 
            '#006400', '#00FF00', '#800000', '#F08080', '#FFFF00', '#000000', '#C0C0C0']
idx = 0
final_avg = []
final_std = []
for exp in results:
    print("exp:", exp)
    reward = []    
    for seed in results[exp]:
        reward.append(list(itertools.chain(*(results[exp][seed]['eval_rewards']))))
    merged = list(itertools.chain.from_iterable(reward))
    out = np.array(merged) 
    out = out.reshape(-1,2)
    df = pd.DataFrame({'Column1':out[:,0],'Column2':out[:,1]})
    length = int(df.Column1.values[-1]/window_size)
    x = np.zeros(length)
    y_avg = np.zeros(length)
    y_std = np.zeros(length)
    for i in range(1, length):
        data = (df[(df.Column1 <= window_size * i) & (df.Column1 > window_size * (i-1))]).Column2.values
        x[i] = window_size * i
        y_avg[i] = np.mean(data)
        y_std[i] = np.std(data)
    avg = y_avg[-10:]
    final_avg.append(np.mean(avg))
    final_std.append(np.std(avg))
    if 'RMSprop' in exp:
        l = 'GAD(RMSprop)'
    elif 'ExtraAdam' in exp:
        l = 'Extra-Adam'
    else:
        l = 'MixedNE-LD(RMSProp)'       
    plt.plot(x, y_avg, color = colors[idx], label = l)
#    if (var):
#        ax.fill_between(x, (y_avg)-(y_std), (y_avg)+(y_std), facecolor=colors[idx], alpha=0.4, interpolate=True)
    idx += 1  
"""
ax.set_title(env_name)    
ax.set_xlabel('Timesteps')
ax.set_ylabel('Reward')
ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))    
"""
#return final_avg, final_std
#avg, std = plot_learning_curves(results, ax, env_name, window_size, var, OnePlayer, best = False)

#ioana code
#new_grad = grad/(func(torch.max(torch.zeros(p.data.size(), device = device), running_loss - c)) + eps_IAKSA)
