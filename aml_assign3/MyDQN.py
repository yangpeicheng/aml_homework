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

MEMORY_SIZE=2000
BATCH_SIZE=32
GAMMA=0.9
INITIAL_EPSILON = 1 # starting value of epsilon
DIS=0.997
FINAL_EPSILON = 0.1 # final value of epsilon
LR=0.005
class MLP(nn.Module):
    def __init__(self):
        in_feature=4
        out_feature=64
        action_num=2
        super(MLP, self).__init__()
        self.input=nn.Linear(in_feature,out_feature)
        self.input.weight.data.normal_(0, 0.1)
        self.out=nn.Linear(out_feature,action_num)
        self.out.weight.data.normal_(0, 0.1)


    def forward(self, input):
        x=self.input(input)
        #x=F.softplus(x)
        x=F.sigmoid(x)
        return self.out(x)

class MyDQN():
    def __init__(self):
        self.N=MEMORY_SIZE
        self.batch_size=BATCH_SIZE
        self.action_num=2

        self.memory=deque()
        self.mlp=MLP()
        self.optimizer=torch.optim.Adam(self.mlp.parameters(),lr=LR)
        self.loss_function=nn.MSELoss()
        self.done = False
        self.epsilon=INITIAL_EPSILON

    def ajust_rl(self,t):
        for param_group in self.optimizer.param_groups:
            param_group['lr']=t


    def perceive(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        if len(self.memory)>self.N:
            self.memory.popleft()
        if len(self.memory)>self.batch_size:
            self.train()


    def train(self):
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
        pre = self.mlp.forward(state_tensor).gather(1, action_tensor.view(-1, 1))
        #print(pre)
        Q=self.mlp.forward(next_state_tensor).detach()
        '''if self.done:
            y=reward_tensor
        else:
            y = torch.max(Q, 1)[0] + reward_tensor'''
        y=GAMMA*torch.max(Q,1)[0]+reward_tensor
        for i in range(self.batch_size):
            if dones[i]:
                y[i]=reward_tensor[i]
                #print(i)
        loss=self.loss_function(pre,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def best_action(self,state):
        state = Variable(torch.FloatTensor(state))
        q = self.mlp.forward(state)
        return np.argmax(q.data.numpy())

    def action(self,state,t):
        #state=Variable(torch.FloatTensor(state))
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))
        #e = max(0.01, 2 - 2 / (1 + math.exp((-t / 30))))
        if np.random.rand()<self.epsilon:
            action=np.random.randint(self.action_num)
        else:
            q = self.mlp.forward(state)
            action=np.argmax(q.data.numpy())
            #action2 = torch.max(q, 1)[1].data.numpy()[0]
            #print("action",action,action2)
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        #self.epsilon*=DIS
        self.epsilon = max(FINAL_EPSILON, self.epsilon)
        return action


def main_test():
    env=gym.make('CartPole-v0')
    env=env.unwrapped
    dqn=MyDQN()
    count=0
    for i in range(1,20000):
        #alpha = max(0.005, min(0.5, 0.5 / (1 + math.exp(((i - 30) / 80)))))
        observation=env.reset()
        #print(alpha)
        #dqn.ajust_rl(alpha)
        c=0
        for j in range(20000):
            #env.render()
            action=dqn.action(observation,i)
            new_observation, reward, done, info = env.step(action)
            c+=1
            r1 = (env.x_threshold - abs(new_observation[0])) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(new_observation[2])) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            #if done:
             #   reward=-10
            dqn.perceive(observation,action,r,new_observation,done)


            observation=new_observation
            if done or j==19999:

                break
            '''
            if done or j==19999:
                if j>5000:
                    count+=30
                elif j>1000:
                    count+=5
                elif j>300:
                    count+=1
                else:
                    count=0
                print(i, j)
                break'''
        #print(i)
        print(i,c)
        #print(i, j + 1)
        if count>100:
            break
    result=[]
    for i in range(2000):
        observation=env.reset()
        for j in range(20000):
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



if __name__=="__main__":
    main_test()

    q=deque()
    q.append(1)
    q.append(4)
    q.append(3)
    q.popleft()
    print(q)