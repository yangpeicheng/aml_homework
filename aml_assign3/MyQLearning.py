import torch
import gym
import numpy as np
import math
def train_CartPole():
    env=gym.make('CartPole-v0')
    episode=20000
    state_n=16
    alpha=0.2
    gamma=0.9
    e=0.1
    Q = np.zeros((state_n, env.action_space.n))
    for i in range(episode):
        env.reset()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        state = observation2state(observation)
        t=1
        while True:
            #env.render()
            t+=1
            if np.random.rand()<e:
                action=env.action_space.sample()
                #print(action)
            else:
                action = np.argmax(Q[state, :])
            observation, reward, done, info = env.step(action)
            new_state=observation2state(observation)
            Q[state,action]=(1-alpha)*Q[state,action]+alpha*(reward+gamma*np.max(Q[new_state,:]))
            state=new_state
            #print(Q)
            #print(reward)
            if done:
                #print(t)
                if t<200:
                    reward=-1
                    Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (
                        reward + gamma * np.max(Q[new_state, :]))
                break
        #print("----")
    print(Q)
def observation2state(observation):
    s=[0 for i in range(4)]
    if observation[0]>-2.4 and observation[0]<2.4:
        s[0]=1
    else:
        s[0]=0
    if observation[1]>0:
        s[1]=1
    else:
        s[1]=0
    if observation[2]>-12 and observation[2]<12:
        s[2]=1
    else:
        s[2]=0
    if observation[3]>0:
        s[3]=1
    else:
        s[3]=0
    return int(bit2num(s))

def bit2num(bits):
    s=0
    for i in range(len(bits)):
        s=s*2+bits[i]
    return s


if __name__=="__main__":
    train_CartPole()