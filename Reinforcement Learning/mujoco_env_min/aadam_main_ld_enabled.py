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

import csv


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

#idk what is thermal noise
#optimizers = ["RMSprop", "ExtraAdam", "SGLD"]

# for walker
"""
env_name = "Walker2d-v4"
optimizers = ['RMSprop__no_noise','ExtraAdam__action_noise_0.3', 'SGLD__thermal_0.01_action_noise_0.01']
action_noises = [False, True, True]
noise_scales = [None, 0.3, 0.01]
epsilon = 1e-2 # i think only for SGLD?
"""

# for halfcheetah
"""
env_name = "HalfCheetah-v4"
optimizers = ['RMSprop__action_noise_0.2', 'ExtraAdam__action_noise_0.01', 'SGLD__thermal_0.01_no_noise']
action_noises = [True, True, False]
noise_scales = [0.2, 0.01, 0]
epsilon = 1e-2
num_steps = 200000
"""

"""
env_name = "Reacher-v4"
optimizers = ['RMSprop/action_noise_0.4', 'ExtraAdam/action_noise_0.2', 'SGLD_thermal_0.001/action_noise_0.2']
action_noises = [True, True, True]
noise_scales = [0.4, 0.2, 0.2]
epsilon = 1e-3
num_steps = 350000
"""

"""
env_name = "Hopper-v4"
optimizers = ['RMSprop__action_noise_0.2', 'ExtraAdam__action_noise_0.3', 'SGLD_thermal_0.001__action_noise_0.2']
action_noises = [True, True, True]
noise_scales = [0.2, 0.3, 0.2]
epsilon = 1e-3
num_steps = 300000
"""

"""
env_name = "InvertedPendulum-v4"
optimizers = ['RMSprop__action_noise_0.1', 'ExtraAdam__action_noise_0.01', 'SGLD__thermal_0.001_action_noise_0.01']
action_noises = [True, True, True]
noise_scales = [0.1, 0.01, 0.01]
epsilon = 1e-3
num_steps = 200000
"""

"""
env_name = "Humanoid-v4"
optimizers_1 = ['RMSprop__no_noise', 'ExtraAdam__action_noise_0.01', 'SGLD_thermal_0.0001__action_noise_0.01']
action_noises = [False, True, True]
noise_scales = [0, 0.01, 0.01]
epsilon = 1e-4
num_steps = 400000
"""

"""
env_name = "Swimmer-v4"
optimizers_2 = ['RMSprop/action_noise_0.4', 'ExtraAdam/action_noise_0.4', 'SGLD_thermal_1e-05/action_noise_0.4']
action_noises = [True, True, True]
noise_scales = [0.4, 0.4, 0.4]
epsilon = 1e-5
num_steps = 300000
"""

"""
env_name = "Ant-v4"
optimizers_3 = ['RMSprop/action_noise_0.4', 'ExtraAdam/action_noise_0.01', 'SGLD_thermal_0.0001/action_noise_0.2']
action_noises = [True, True, True]
noise_scales = [0.4, 0.01, 0.2]
epsilon = 1e-4
num_steps = 500000 #wait the graph shows 100 000
"""

"""
env_name = "HalfCheetah-v4"
optimizers = ["SGLMLD"]
optimizers = ["SGLD"]
action_noises = [False]
noise_scales = [0]
epsilon = 1e-4
num_steps = 200000
"""

# environments = ["Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]
# step_list = [300000, 200000, 300000]
# action_noises = [True, False, True]
# noise_scales = [0.01, 0, 0.2]
# epsilon_array = [1e-2, 1e-2, 1e-3]

environments = ["Walker2d-v4", "HalfCheetah-v4", "Hopper-v4", "Ant-v4", "Swimmer-v4", "Reacher-v4", "InvertedPendulum-v4", "Humanoid-v4"]
step_list = [300000, 200000, 300000, 100000, 300000, 300000, 200000, 300000]
action_noises = [True, False, True, True, True, True, True, True]
noise_scales = [0.01, 0, 0.2, 0.2, 0.4, 0.2, 0.01, 0.01]
epsilon_array = [1e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-3, 1e-3, 1e-4]


#fixed_c_array = [-100,-100,-150,-150]
#fixed_c_array = [0,0,-200,-200]
for j in range(len(environments)):
    

    if j < 3:
        continue   
    # if j == 0:
    #     continue
    # # if j == 1:
    # #     continue
    # if j == 2:
    #     continue
    # if j == 3:
    #     continue
    if j == 7:
        continue
    # if j == 5:
    #     continue
    # if j == 6:
    #     continue

    
    # if j == 5:
    #     continue

    optimizer = "SGLMLD"
    epsilon = epsilon_array[j]
    env_name = environments[j]
    num_steps = step_list[j]
    action_noise = action_noises[j]
    noise_scale = noise_scales[j]
    #fixed_c = fixed_c_array[j]
    fixed_c = None
    print(env_name)
    print(num_steps)

    for i in range(0,1):
        for seed in range(2):
            env = NormalizedActions(gym.make(env_name))
            eval_env = NormalizedActions(gym.make(env_name))

            agent = DDPG(beta=beta, epsilon=epsilon, learning_rate=learning_rate, gamma=gamma, 
                tau=tau, hidden_size_dim0=hidden_size_dim0, hidden_size_dim1=hidden_size_dim1, 
                num_inputs=env.observation_space.shape[0], action_space=env.action_space, 
                train_mode=True, alpha=alpha, replay_size=replay_size, optimizer=optimizer, 
                two_player=True)

            results_dict = {'eval_rewards': [],
                        'value_losses': [],
                        'policy_losses': [],
                        'adversary_losses': [],
                        'train_rewards': []
                        }
            value_losses = []
            policy_losses = []
            adversary_losses = []


            base_dir = os.getcwd() + '/my_test/' + env_name +'/'+ optimizer + '/'
            print(base_dir)

            run_number = 0
            while os.path.exists(base_dir + str(run_number)):
                run_number += 1
            base_dir = base_dir + str(run_number)

            os.makedirs(base_dir)
            normalnoise = NormalActionNoise(mu=np.zeros(env.action_space.shape[0]),
                                            sigma=float(noise_scale) * np.ones(env.action_space.shape[0])
                                            ) if action_noise else None
            def reset_noise(a, a_noise):
                if a_noise is not None:
                    a_noise.reset()

            total_steps = 0
            print(base_dir)

            if num_steps is not None:
                assert num_epochs is None
                nb_epochs = int(num_steps) // (num_epochs_cycles * num_rollout_steps)
            else:
                nb_epochs = 500

            state = agent.Tensor([env.reset()])
            eval_state = agent.Tensor([eval_env.reset()])
            eval_reward = 0
            episode_reward = 0
            agent.train()
            reset_noise(agent, normalnoise)
            if visualize:
                vis = visdom.Visdom(env=base_dir)
            else:
                vis = None

            
            train_steps = 0
            ratio = ratio + 1
            policy_loss_min = 9999999
            for epoch in trange(nb_epochs):
                for cycle in range(num_epochs_cycles):
                    curr_policy_loss = 0
                    with torch.no_grad():
                        # training stage
                        for t_rollout in range(num_rollout_steps):
                            action = agent.select_action(state, action_noise=normalnoise, mdp_type=exploration_method)
                            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                            #print("reward: ", reward)
                            total_steps += 1
                            episode_reward += reward

                            action = agent.Tensor(action)
                            mask = agent.Tensor([not done])
                            next_state = agent.Tensor([next_state])
                            reward = agent.Tensor([reward])
                            agent.store_transition(state, action, mask, next_state, reward)

                            state = next_state

                            if done:
                                #print("episode_reward: ", episode_reward)
                                #print("total_steps: ", total_steps)
                                results_dict['train_rewards'].append((total_steps, np.mean(episode_reward)))
                                episode_reward = 0
                                state = agent.Tensor([env.reset()])
                                reset_noise(agent, normalnoise)

                    if len(agent.memory) > batch_size:
                        # update the parameters
                        for t_train in range(number_of_train_steps):
                            warmup = (math.floor(np.power(1 + 1e-5, train_steps)))
                            # warmup steps for SGLD + two_player
                            if ((optimizer == 'SGLD' or optimizer == 'SGLMLD') and two_player):
                                kt = np.minimum(Kt, warmup)                 
                                agent.initialize()
                            # noram setup for RMSPROP and SGLD + one_player
                            else:
                                kt = 1
                    
                            for k in range(kt):
                                sgld_outer_update = (k == kt - 1)
                                if fixed_c != None:
                                    policy_loss_min = fixed_c
                                else:
                                    policy_loss_min = min(policy_loss_min, curr_policy_loss)
                                #c_diff_policy = curr_policy_loss - policy_loss_min
                                # print("policy_loss_min: ", policy_loss_min)
                                # print("curr_policy_loss: ", curr_policy_loss)
                                # print("==")
                                
                                if optimizer != 'SGLMLD':
                                    c_difference = 0
                                value_loss, policy_loss, adversary_loss = agent.update_parameters(
                                        batch_size=batch_size,
                                        sgld_outer_update=sgld_outer_update,
                                        mdp_type=method,
                                        exploration_method=exploration_method,
                                        policy_loss_min=policy_loss_min)
                                value_losses.append(value_loss)
                                policy_losses.append(policy_loss)
                                adversary_losses.append(adversary_loss)
                                curr_policy_loss = policy_loss
                            train_steps += 1
                        
                        results_dict['value_losses'].append((total_steps, np.mean(value_losses)))
                        results_dict['policy_losses'].append((total_steps, np.mean(policy_losses)))
                        results_dict['adversary_losses'].append((total_steps, np.mean(adversary_losses)))
                        del value_losses[:]
                        del policy_losses[:]
                        del adversary_losses[:]
                    with torch.no_grad():
                        # evaluation stage, with different environment from training stage
                        for t_rollout in range(num_rollout_steps):
                            action = agent.select_action(eval_state, mdp_type='mdp')

                            next_eval_state, reward, done, _ = eval_env.step(action.cpu().numpy()[0])
                            eval_reward += reward

                            next_eval_state = agent.Tensor([next_eval_state])

                            eval_state = next_eval_state
                            if done:
                                results_dict['eval_rewards'].append((total_steps, eval_reward))
                                #print(eval_reward)
                                eval_state = agent.Tensor([eval_env.reset()])
                                eval_reward = 0
                    # save the model 
                    save_model(actor=agent.actor, adversary=agent.adversary, basedir=base_dir, obs_rms=agent.obs_rms,
                        rew_rms=agent.ret_rms)
                    with open(base_dir + '/results', 'wb') as f:
                        pickle.dump(results_dict, f)


                    vis_plot(vis, results_dict)

            with open(base_dir + '/results', 'wb') as f:
                pickle.dump(results_dict, f)
            save_model(actor=agent.actor, adversary=agent.adversary, basedir=base_dir, obs_rms=agent.obs_rms, rew_rms=agent.ret_rms)

            env.close()
