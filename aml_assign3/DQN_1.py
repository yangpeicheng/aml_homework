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
GAMMA=0.99
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
LR=0.0025
HIDDEN_SIZE=64

'''

mountaincar
MEMORY_SIZE=2000
BATCH_SIZE=32
GAMMA=0.99
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
LR=0.005

acrobot 参数
MEMORY_SIZE=2000
BATCH_SIZE=32
GAMMA=0.9
INITIAL_EPSILON = 1 # starting value of epsilon
DIS=0.997
FINAL_EPSILON = 0.1 # final value of epsilon
LR=0.005
'''

def config(id):
    global FINAL_EPSILON,LR,HIDDEN_SIZE
    if id=="CartPole-v0":
        FINAL_EPSILON=0.001
        LR=0.0025
    elif id=="MountainCar-v0":
        FINAL_EPSILON=0.01
        LR=0.005
    else:
        FINAL_EPSILON=0.1
        LR=0.005

class MLP(nn.Module):
    def __init__(self,env):
        in_feature=len(env.observation_space.low)
        hidden_feature=HIDDEN_SIZE
        action_num=env.action_space.n
        super(MLP, self).__init__()
        self.input=nn.Linear(in_feature,hidden_feature)
        self.input.weight.data.normal_(0, 0.1)
        self.out=nn.Linear(hidden_feature,action_num)
        self.out.weight.data.normal_(0, 0.1)


    def forward(self, input):
        x=self.input(input)
        #x=F.softplus(x)
        x=F.sigmoid(x)
        #x=F.relu(x)
        return self.out(x)

class MyDQN():
    def __init__(self,env):
        self.N=MEMORY_SIZE
        self.batch_size=BATCH_SIZE
        self.action_num=env.action_space.n

        self.memory=deque()
        self.mlp=MLP(env)
        self.optimizer=torch.optim.Adam(self.mlp.parameters(),lr=LR)
        self.loss_function=nn.MSELoss()
        self.done = False
        self.epsilon=INITIAL_EPSILON
        self.loss = 0

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
        self.loss += float(loss.data)
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

    def get_loss(self):
        result=self.loss
        self.loss=0
        return result

def main_test(id):
    config(id)
    env=gym.make(id)
    env=env.unwrapped
    dqn=MyDQN(env)
    if id=='CartPole-v0':
        T=20000
    else:
        T=2000

    count = 0
    train_result=[]
    train_loss=[]
    for i in range(500):
        observation=env.reset()
        for j in range(T):
            action=dqn.action(observation,i)
            new_observation, reward, done, info = env.step(action)
            if id=='CartPole-v0' :
                r1 = (env.x_threshold - abs(new_observation[0])) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(new_observation[2])) / env.theta_threshold_radians - 0.5
                reward = r1 + r2
                '''if j<2000:
                    reward=-200'''

            elif  done:
                reward=100
            dqn.perceive(observation,action,reward,new_observation,done)
            observation=new_observation
            if done==False and j!=T-1:
                continue
            train_result.append(j)

            if id=='CartPole-v0':
                if done or j==T-1:
                    if j > 5000:
                        count += 1
                    else:
                        count = 0
                    print(i, j)
                    break
            elif id=='MountainCar-v0':
                print(i, j)
                if done and j<500:
                    count+=1
                else:
                    count=0
                break
            else:
                print(i, j)
                if done and j<500:
                    count+=1
                else:
                    count=0
                break
        train_loss.append(dqn.get_loss()/train_result[-1])
        if id=='CartPole-v0' and count>=5:
            break
        if id!='CartPole-v0' and count>=100:
            break
    print(train_loss)
    print(train_result)
    plt.plot(train_loss)
    plt.xlabel("round")
    plt.ylabel("loss")
    plt.show()
    if id!='CartPole-v0':
        train_result = -np.array(train_result)
    plt.plot(train_result)
    plt.xlabel("round")
    plt.ylabel("reward")
    plt.show()


    result=[]
    for i in range(200):
        observation=env.reset()
        for j in range(T):
            #env.render()
            action=dqn.best_action(observation)
            observation, reward, done, info = env.step(action)
            if done or j == T-1:
                print("test",j+1)
                result.append(j+1)
                break
    result=np.array(result)
    if id!='CartPole-v0':
        result=-result
    plt.plot(result)
    plt.xlabel("round")
    plt.ylabel("reward")
    plt.show()
    print("mean",np.mean(result))
    print("var",np.std(result))
    print("len",len(result))




if __name__=="__main__":
    #main_test('Acrobot-v1')
    #main_test('MountainCar-v0')
    main_test('CartPole-v0')