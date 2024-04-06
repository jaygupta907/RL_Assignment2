import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
from collections import deque
import numpy as np
import random
from torch import optim
import sys
from torch.autograd import Variable
import matplotlib.pyplot as plt

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print(device)



#Defining Dueling Deep Q Network
class Dueling_DQN(nn.Module):
    def __init__(self,environment,Advantage_type='average'):
        super(Dueling_DQN, self).__init__()
        self.state_dim = environment.observation_space.shape[0]
        self.action_dim = environment.action_space.n
        self.feature_dim = 256
        self.Advantage_type =  Advantage_type
        self.Feature_net = nn.Sequential(
            nn.Linear(self.state_dim,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,self.feature_dim),
            nn.ReLU()
        )
        self.Value_net = nn.Sequential(
            nn.Linear(self.feature_dim,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        self.Advantage_net= nn.Sequential(
            nn.Linear(self.feature_dim,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,self.action_dim)

        )
        self.max_layer = nn.MaxPool1d(self.action_dim)
    def forward(self,state):
        x = self.Feature_net(state)
        state_value = self.Value_net(x)
        Advantage = self.Advantage_net(x)
        if self.Advantage_type=='average':
            Action_value = state_value + (Advantage-torch.mean(Advantage,dim=1,keepdim=True)) 
        else :
            Action_value = state_value + (Advantage-self.max_layer(Advantage))
        return Action_value




#Defining Replay Buffer
class Replay_Buffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)
    



class Dueling_DQN_agent:
    def __init__(self,environment,buffer_size=50000,gamma=0.99,learning_rate=0.0003,initial_epsilon=0.3,
                 final_epsilon =0.001,decay_time=100,tau=0.05,type='average'):

        self.environment=environment
        self.type=type
        self.replay_buffer = Replay_Buffer(buffer_size)
        self.Dueling_DQN_online = Dueling_DQN(self.environment,self.type).to(device)
        self.Dueling_DQN_target = Dueling_DQN(self.environment,self.type).to(device)
        self.criterion  = nn.MSELoss()
        self.optimizer  = optim.Adam(self.Dueling_DQN_online.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.inital_epsilon = initial_epsilon
        self.final_epsilon  = final_epsilon
        self.decay_time = decay_time
        self.tau = tau
        for target_param, param in zip(self.Dueling_DQN_target.parameters(), self.Dueling_DQN_online.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        

    def get_action(self,state,t):
        p=random.random()
        epsilon = self.inital_epsilon - (self.inital_epsilon-self.final_epsilon)*(t/self.decay_time)
        if p<epsilon:
            action = random.randint(0,1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = torch.argmax(self.Dueling_DQN_online(state),dim=1)
            action =  action.item()
        return action

    def update(self,batch_size):
        states, actions, rewards, next_states, done = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        done  =  torch.FloatTensor(done).unsqueeze(1).to(device)

        online_Q_next = self.Dueling_DQN_online(next_states)
        target_Q_next = self.Dueling_DQN_target(next_states)
        online_max_action = torch.argmax(online_Q_next,dim=1,keepdim=True)
        target = rewards + (1-done)*self.gamma*target_Q_next.gather(1,online_max_action.long())
        self.optimizer.zero_grad()

        Q_value = self.Dueling_DQN_online(states).gather(1,actions.long())
        loss = self.criterion(target,Q_value)
        loss.backward()
        self.optimizer.step()
        for target_param, param in zip(self.Dueling_DQN_target.parameters(), self.Dueling_DQN_online.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
        return loss.item()




def train(num_expts,num_episodes,num_timesteps,type):
    rewards_runs = []
    loss_runs=[]
    for expt in range(num_expts):
        sys.stdout.write("Experiment Number:{}\n".format(expt+1))
        environment = gym.make("Cartpole-v1")
        agent = Dueling_DQN_agent(environment,type=type)
        rewards=[]
        losses=[]
        avg_rewards=[]
        avg_loss =[]
        batch_size = 128
        for episode  in range(num_episodes):
            episode_reward=[]
            episode_loss = []
            state = environment.reset()
            environment.render(mode='human')
            for j in range(num_timesteps):
                loss=0
                action = agent.get_action(state,episode)
                next_state, reward, done, info = environment.step(action)
                environment.render(mode='human')
                agent.replay_buffer.push(state, action, reward, next_state, done)
                if len(agent.replay_buffer)>batch_size:
                    loss = agent.update(batch_size)
                state = next_state
                episode_reward.append(reward)
                episode_loss.append(loss)
                if done:
                    sys.stdout.write("episode: {}, reward: {}\n".format(episode+1, np.round(np.sum(episode_reward), decimals=3)))
                    break
            rewards.append(np.sum(episode_reward))
            losses.append(np.mean(episode_loss))
            avg_rewards.append(np.mean(rewards[-10:]))
            avg_loss.append(np.mean(losses[-10:]))
        rewards_runs.append(avg_rewards)
        loss_runs.append(avg_loss)
    return rewards_runs,loss_runs


num_expts = 5
num_episodes =  100
num_timesteps = 500
episodes = [i+1 for i in range(num_episodes)]


rewards_avg,loss_avg = train(num_expts,num_episodes,num_timesteps,'average')
rewards_max,loss_max = train(num_expts,num_episodes,num_timesteps,'max')


rewards_avg_std = np.std(rewards_avg,axis=0)
rewards_avg_mean = np.mean(rewards_avg,axis=0)
plt.plot(episodes,rewards_avg_mean,color='red')
plt.fill_between(episodes,rewards_avg_mean-rewards_avg_std,rewards_avg_mean+rewards_avg_std,alpha=0.5,facecolor='red')
plt.plot()
plt.xlabel('Episode Number')
plt.ylabel('Episode Reward (Type-I)')
plt.grid()
plt.show()


rewards_max_std = np.std(rewards_max,axis=0)
rewards_max_mean = np.mean(rewards_max,axis=0)
plt.plot(episodes,rewards_max_mean,color='red')
plt.fill_between(episodes,rewards_max_mean-rewards_max_std,rewards_max_mean+rewards_max_std,alpha=0.5,facecolor='red')
plt.plot()
plt.xlabel('Episode Number')
plt.ylabel('Episode Reward (Type-II)')
plt.grid()
plt.show()


plt.plot(episodes,rewards_avg_mean,color='green')
plt.plot(episodes,rewards_max_mean,color='red')
plt.fill_between(episodes,rewards_avg_mean-rewards_avg_std,rewards_avg_mean+rewards_avg_std,alpha=0.5,facecolor='green')
plt.fill_between(episodes,rewards_max_mean-rewards_max_std,rewards_max_mean+rewards_max_std,alpha=0.5,facecolor='red')
plt.legend(["Type-I","Type-II"])
plt.plot()
plt.xlabel('Episode Number')
plt.ylabel('Episode Reward')
plt.grid()
plt.show()

np.savetxt("Cartpole_avg_mean.csv",rewards_avg_mean,delimiter=',')
np.savetxt("Cartpole_avg_std.csv",rewards_avg_std,delimiter=',')
np.savetxt("Cartpole_max_mean.csv",rewards_max_mean,delimiter=',')
np.savetxt("Cartpole_max_std.csv",rewards_max_std,delimiter=',')