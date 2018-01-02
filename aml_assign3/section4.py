import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import deque
import math
import numpy as np
import random
import gym
from matplotlib import pyplot as plt


MEMORY_SIZE=10000
BATCH_SIZE=32
GAMMA=0.95
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon

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
        self.optimizer=torch.optim.Adam(self.mlp.parameters(),lr=0.001)
        self.loss_function=nn.MSELoss()
        self.step=500
        self.current=1
        self.epsilon=INITIAL_EPSILON


    def ajust_rl(self,t):
        for param_group in self.optimizer.param_groups:
            param_group['lr']=t


    def perceive(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        if len(self.memory)>self.N:
            self.memory.popleft()
        if len(self.memory)>1000:
            self.train()

    def best_action(self,state):
        state = Variable(torch.FloatTensor(state))
        q = self.mlp.forward(state)
        return np.argmax(q.data.numpy())


    def train(self):
        self.current+=1

        if self.current % self.step == 0:
            self.target.load_state_dict(self.mlp.state_dict())

        batch_range=random.sample(self.memory,self.batch_size)
        states=[]
        actions=[]
        rewards=[]
        next_states=[]
        dones=[]
        for data in batch_range:
            states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])
            dones.append(data[4])
        state_tensor=Variable(torch.FloatTensor(states))
        action_tensor=Variable(torch.LongTensor(np.array(actions)))
        reward_tensor=Variable(torch.FloatTensor(rewards))
        next_state_tensor=Variable(torch.FloatTensor(next_states))


        Q=self.target.forward(next_state_tensor)

        y=torch.max(Q,1)[0]+reward_tensor
        for i in range(self.batch_size):
            if dones[i]:
                y[i]=reward_tensor[i]
                #print(i)
        pre=self.mlp.forward(state_tensor).gather(1,action_tensor.view(-1,1))
        loss=self.loss_function(pre,y.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def action(self,state,t):
        state=Variable(torch.FloatTensor(state))
        q=self.mlp.forward(state)
        e = max(0.01, 2 - 2 / (1 + math.exp((-t / 30))))
        #e=0.5/math.sqrt(t)
        if np.random.rand()<e:
            action=np.random.randint(self.action_num)
        else:
            action=np.argmax(q.data.numpy())
        self.epsilon-=(INITIAL_EPSILON-FINAL_EPSILON)/10000
        return action


def main_test():
    env=gym.make('CartPole-v0')
    env=env.unwrapped
    dqn=MyDQN()
    count = 0
    for i in range(3000):
        #alpha = max(0.005, min(0.5, 0.5 / (1 + math.exp(((i - 30) / 80)))))
        observation=env.reset()
        #print(alpha)
        #dqn.ajust_rl(alpha)

        flag=False
        for j in range(20000):
            action=dqn.action(observation,i)
            new_observation, reward, done, info = env.step(action)


            #if done:
             #   reward=-1
            dqn.perceive(observation,action,reward,new_observation,done)


            observation=new_observation
            if done or j == 19999:
                if j>5000:
                    count+=1
                else:
                    count=0
                print(i,j)
                break
        #print("s:",i,count)
        if count>=5:
            break

    result=[]
    for i in range(2000):
        observation=env.reset()
        for j in range(20000):
            env.render()
            action=dqn.best_action(observation)
            observation, reward, done, info = env.step(action)
            if done or j == 19999:
                print("test",j)
                result.append(j)
                break
    plt.plot(result)
    plt.show()
    print("mean",np.mean(result))
    print("var",np.std(result))
    print("len",len(result))

main_test()
