import gym
import numpy as np
import math
import random


def train(id='CartPole-v0'):
    env=gym.make(id)
    env=env.unwrapped
    bounds,state_size=getBounds_Statesize(id)
    if id=='CartPole-v0':
        train_size=20000
    else:
        train_size=2000
    episode=2000
    gamma=0.99
    streak=0
    Q = np.zeros(state_size+(env.action_space.n,))
    for i in range(episode):
        e=max(0.01, 2-2/(1+math.exp((-i/30))))
        alpha = max(0.1, min(0.5, 1 / (1 + math.exp(((i - 30) / 80)))))
        observation=env.reset()
        state = observation2state(observation,state_size,bounds)
        flag=False
        for j in range(train_size):
            #env.render()
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
                if id!='CartPole-v0':
                    flag=True
                    streak+=1
                    print("finish",j)
                    break
                else:
                    if j>500:
                        streak+=1
                        flag=True
                        print(j)
                    break
        if flag==False:
            streak=0
        if streak>100:
            break
    print(Q)
    return Q

def test(id):
    q = train(id)
    env = gym.make(id)
    env = env.unwrapped
    bounds, state_size = getBounds_Statesize(id)
    episode=2000
    if id=='CartPole-v0':
        train_size=20000
    else:
        train_size=2000
    result=[]
    for i in range(episode):
        observation=env.reset()
        state = observation2state(observation,state_size,bounds)
        for j in range(train_size):
            #env.render()
            action = np.argmax(q[state])
            observation, reward, done, info = env.step(action)
            new_state=observation2state(observation,state_size,bounds)
            state=new_state
            if done:
                result.append(j)
                break
    print("mean",np.mean(result))
    print("var",np.std(result))
    print("len",len(result))

def getBounds_Statesize(id):
    env=gym.make(id)
    bounds=list(zip(env.observation_space.low,env.observation_space.high))
    if id=='CartPole-v0':
        bounds[1]=(-1.5,1.5)
        bounds[3]=(-1.5,1.5)
        state_size=(1,1,6,5)
    elif id=='MountainCar-v0':
        state_size = (5, 2)
    elif id=='Acrobot-v1':
        state_size = (1, 1, 1, 1, 2, 2)
    return bounds,state_size



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
    test('CartPole-v0')