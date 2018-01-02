import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import deque
import math
import numpy as np
import random
import gym

MEMORY_SIZE=10000
BATCH_SIZE=32
GAMMA=0.95

class MLP(nn.Module):
    def __init__(self):
        in_feature=4
        out_feature=100
        action_num=2
        super(MLP, self).__init__()
        self.input=nn.Linear(in_feature,out_feature)
        #self.input.weight.data.normal_(0, 0.1)
        self.out=nn.Linear(out_feature,action_num)
        #self.out.weight.data.normal_(0, 0.1)


    def forward(self, input):
        x=self.input(input)
        #x=F.softplus(x)
        x=F.relu(x)
        return self.out(x)

class MyDQN():
    def __init__(self):
        self.N=MEMORY_SIZE
        self.batch_size=BATCH_SIZE
        self.action_num=2

        self.memory=deque()
        self.mlp=MLP()
        self.target=MLP()
        self.optimizer=torch.optim.Adam(self.mlp.parameters(),lr=0.01)
        self.loss_function=nn.MSELoss()
        self.step=100
        self.current=1
        self.loss_value=0

    def ajust_rl(self,t):
        for param_group in self.optimizer.param_groups:
            param_group['lr']=t


    def perceive(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state))
        self.done=done
        if len(self.memory)>self.N:
            self.memory.popleft()
        if len(self.memory)>self.batch_size:
            self.train()


    def train(self):
        self.current+=1

        if self.current % self.step == 0:
            self.target.load_state_dict(self.mlp.state_dict())

        batch_range=random.sample(self.memory,self.batch_size)
        states=[]
        actions=[]
        rewards=[]
        next_states=[]
        for data in batch_range:
            states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])
        state_tensor=Variable(torch.FloatTensor(states))
        action_tensor=Variable(torch.LongTensor(np.array(actions)))
        reward_tensor=Variable(torch.FloatTensor(rewards))
        next_state_tensor=Variable(torch.FloatTensor(next_states))


        Q=self.target.forward(next_state_tensor)

        if self.done:
            y=reward_tensor
        else:
            y = torch.max(Q, 1)[0] + reward_tensor
        pre=self.mlp.forward(state_tensor).gather(1,action_tensor.view(-1,1))
        loss=self.loss_function(pre,y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_value=loss

    def action(self,state,t):
        state=Variable(torch.FloatTensor(state))
        q=self.mlp.forward(state)
        e = max(0.01, 2 - 2 / (1 + math.exp((-t / 30))))
        #e=0.5/math.sqrt(t)
        if np.random.rand()<e:
            action=np.random.randint(self.action_num)
        else:
            action=np.argmax(q.data.numpy())
        return action

    def get_loss(self):
        return self.loss_value

def main_test():
    env=gym.make('CartPole-v0')
    env=env.unwrapped
    dqn=MyDQN()
    for i in range(1,3000):
        #alpha = max(0.005, min(0.5, 0.5 / (1 + math.exp(((i - 30) / 80)))))
        observation=env.reset()
        #print(alpha)
        #dqn.ajust_rl(alpha)
        for j in range(20000):
            action=dqn.action(observation,i)
            new_observation, reward, done, info = env.step(action)


            r1 = (env.x_threshold - abs(new_observation[0])) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(new_observation[2])) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            dqn.perceive(observation,action,r,new_observation,done)


            observation=new_observation
            if done or j == 19999:
                print(i,j)
                break


main_test()
