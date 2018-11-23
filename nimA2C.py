import os
import numpy as np
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optimizers as optim
import torch.autograd as autograd
from torch.autograd import Variable

SEED = 2
N_ACTIONS = 2
N_INPUTS = 4

limit = 4 ###

# env = gym.make('CartPole-v0')
# env.observation_space.high

GAMMA = .95
LR = 3e-3
N_GAMES = 1000
N_STEPS = 20

# run_set(N_GAMES, N_STEPS, LR, GAMMA)

lrs = [8e-3, 3e-3]
eps = [.95]
n_steps = [5, 10, 20]
#EXP_NAME = "Cartpole_nstep_gridsearch_LR_EPS_NSTEPS_121617_b"
#os.rmdir("experiments/"+EXP_NAME)
#os.mkdir("experiments/"+EXP_NAME)
# for lr in lrs:
#     for e in eps:
#         for ns in n_steps:  
#             try: run_set(2000, ns, lr, e)
#             except: print("Failed to run set: NS "+str(ns)+" lr "+str(lr)+" eps "+str(e))

def get_tuple_from_id(id):
	return (id/(2**limit), id%(2**limit))

def get_id_from_tuple(move):
	return move[0]*(2**limit) + move[1]

# def run_set(N_GAMES, N_STEPS, LR, EPS):
#     # env = gym.make('CartPole-v0')

#     model = ActorCritic()
#     optimizer = optim.Adam(model.parameters(), lr=LR)
#     state = game.get_board() #TBD

#     finished_games = 0
#     num_games = []
#     scores = []
#     value_losses = []
#     action_gains = []
    
#     already_logged = False
    
#     states = []
#     actions = []
#     rewards = []
#     dones = []
#     cum_scores = []
    
#     game_current_score = 0

#     while finished_games < N_GAMES:

#         del states[:]
#         del actions[:]
#         del rewards[:]
#         del dones[:]
#         del cum_scores[:]

#         # act phase
#         for i in range(N_STEPS):
#             s = torch.from_numpy(state).float().unsqueeze(0)

#             action_probs = model.get_action_probs(Variable(s))
#             action = action_probs.multinomial().data[0][0]
#             game.play_move(get_tuple_from_id(action))
#             # next_state = 
#             # next_state, reward, done, _ = env.step(action)

#             states.append(state)
#             actions.append(action)
#             rewards.append(reward)
#             dones.append(done)
            
#             game_current_score += 1
#             cum_scores.append(game_current_score)

#             if done: 
#                 game_current_score = 0
#                 state = env.reset()
#                 finished_games += 1
#                 already_logged = False

#             else: state = next_state

#         # only taking windows in which failure occurs
#         if True in dones and not 200 in cum_scores:
#             # Reflect phase

#             R = []
#             rr = rewards
#             rr.reverse()

#             if dones[-1] == True:
#                 next_return = -30
#             else:
#                 s = torch.from_numpy(states[-1]).float().unsqueeze(0)
#                 next_return = model.get_state_value(Variable(s)).data[0][0]

#             R.append(next_return)
#             dones.reverse()
#             for r in range(1, len(rr)):
#                 if not dones[r]:
#                     this_return = rr[r] + next_return * EPS
#                 else:
#                     this_return = -30

#                 R.append(this_return)
#                 next_return = this_return

#             R.reverse()
#             dones.reverse()
#             rewards = R

#             s = Variable(torch.FloatTensor(states))

#             action_probs, state_values = model.evaluate_actions(s)

#             action_log_probs = action_probs.log() 

#             advantages = Variable(torch.FloatTensor(rewards)).unsqueeze(1) - state_values

#             entropy = (action_probs * action_log_probs).sum(1).mean()

#             a = Variable(torch.LongTensor(actions).view(-1,1))

#             chosen_action_log_probs = action_log_probs.gather(1, a)

#             action_gain = (chosen_action_log_probs * advantages).mean()

#             value_loss = advantages.pow(2).mean()

#             total_loss = value_loss - action_gain - 0.0001*entropy

#             optimizer.zero_grad()

#             total_loss.backward()

#             nn.utils.clip_grad_norm(model.parameters(), 0.5)

#             optimizer.step()

#         if finished_games % 50 == 0 and not already_logged:
#             try:
#                 s = test_model(model)
#                 scores.append(s)
#                 num_games.append(finished_games)
#                 action_gains.append(action_gain.data.numpy()[0])
#                 value_losses.append(value_loss.data.numpy()[0])
#                 already_logged = True
#             except:
#                 continue

#     EXP = "Cartpole_nstep_"+"LR_"+str(LR)+"_N_STEPS_"+str(N_STEPS)+"_EPS_"+str(EPS)+".png"

#     plt.plot(num_games, scores)
#     plt.xlabel("N_GAMES")
#     plt.ylabel("Score")
#     plt.title(EXP)
#     plt.show()
    
#     plt.plot(num_games, value_losses)
#     plt.xlabel("N_GAMES")
#     plt.ylabel("Value loss")
#     plt.show()
    
#     plt.plot(num_games, action_gains)
#     plt.xlabel("N_GAMES")
#     plt.ylabel("action gains")
#     plt.show()

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.linear1 = nn.Linear(N_INPUTS, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        
        self.actor = nn.Linear(64, N_ACTIONS)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        
        x = self.linear2(x)
        x = F.relu(x)
        
        x = self.linear3(x)
        x = F.relu(x)
        
        return x
    
    def get_action_probs(self, x):
        x = self(x)
        action_probs = F.softmax(self.actor(x))
        return action_probs
    
    def get_state_value(self, x):
        x = self(x)
        state_value = self.critic(x)
        return state_value
    
    def evaluate_actions(self, x):
        x = self(x)
        action_probs = F.softmax(self.actor(x))
        state_values = self.critic(x)
        return action_probs, state_values

