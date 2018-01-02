import gym
import numpy as np
import math
import random

def train_CartPole():
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    bounds=list(zip(env.observation_space.low,env.observation_space.high))
    bounds[1]=(-1.5,1.5)
    bounds[3]=(-1.5,1.5)
    state_size=(1,1,6,5)
    episode=2000
    gamma=0.99
    streak=0
    error=5
    Q = np.zeros(state_size+(env.action_space.n,))
    for i in range(episode):
        e=max(0.01, 2-2/(1+math.exp((-i/30))))
        alpha = max(0.05, min(0.5, 1 / (1 + math.exp(((i - 30) / 80)))))
        observation=env.reset()
        state = observation2state(observation,state_size,bounds)
        t=0
        for j in range(20000):
            t+=1
            r=random.random()
            if r<e:
                action=env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            observation, reward, done, info = env.step(action)
            new_state=observation2state(observation,state_size,bounds)
            Q[state+(action,)]=(1-alpha)*Q[state+(action,)]+alpha*(reward+gamma*np.amax(Q[new_state]))
            state=new_state
            if done:
                print(t)
                if t>800:
                    streak+=1
                elif error>0:
                    error-=1
                else:
                    streak=0
                    error=3
                break
        if streak>=200:
            print("finish")
            break
    print(Q)
    return Q

def testQ():
    q=train_CartPole()
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    episode=2000
    bounds=list(zip(env.observation_space.low,env.observation_space.high))
    bounds[1]=(-1.5,1.5)
    bounds[3]=(-1.5,1.5)
    state_size=(1,1,6,5)
    result=[]
    for i in range(episode):
        observation=env.reset()
        state = observation2state(observation,state_size,bounds)
        t=0
        while True:
            #env.render()
            t+=1
            action = np.argmax(q[state])
            observation, reward, done, info = env.step(action)
            new_state=observation2state(observation,state_size,bounds)
            state=new_state
            if done:
                result.append(t)
                #print(t)
                break
    print("mean",np.mean(result))
    print("var",np.std(result))

def observation2state(observation,state_size,bounds):
    state=[]
    for i in range(len(observation)):
        if observation[i]<bounds[i][0]:
            state.append(0)
        elif observation[i]>bounds[i][1]:
            state.append(state_size[i]-1)
        else:
            bound_width = bounds[i][1] - bounds[i][0]
            offset = (state_size[i]-1)*bounds[i][0]/bound_width
            scaling = (state_size[i]-1)/bound_width
            state.append(int(round(scaling*observation[i] - offset)))
    return tuple(state)



if __name__=="__main__":
    testQ()
