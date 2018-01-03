import gym
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from gym.wrappers import Monitor


RECORD=False


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
    count=0
    Q = np.zeros(state_size+(env.action_space.n,))
    for i in range(episode):
        e=max(0.01, 2-2/(1+math.exp((-i/30))))
        #e=max(0.01,(1-i/episode))
        alpha = max(0.1, min(0.5, 1 / (1 + math.exp(((i - 30) / 80)))))
        #alpha=0.01
        observation=env.reset()
        state = observation2state(observation,state_size,bounds)
        for j in range(train_size):
            #env.render()
            #print(j)
            r=random.random()
            if r<e:
                action=env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            observation, reward, done, info = env.step(action)
            new_state=observation2state(observation,state_size,bounds)
            #print(observation)
            #print(new_state)
            Q[state+(action,)]=(1-alpha)*Q[state+(action,)]+alpha*(reward+gamma*np.amax(Q[new_state]))
            state=new_state
            #if j==train_size-1 and id=='CartPole-v0':
            #    done=True

            if id == 'CartPole-v0':
                if done or j == train_size - 1:
                    if j > 500:
                        count += 1
                    else:
                        count = 0
                    print(i, j)
                    break
            elif id == 'MountainCar-v0' and done:
                if j < 500:
                    count += 1
                else:
                    count = 0
                print(i, j)
                break
            elif done:
                if j < 150:
                    count += 1
                else:
                    count = 0
                print(i, j)
                break
        #print("count",i,count)
        if  count >= 100:
            break
    #print(Q)
    return Q

def test(id):
    q = train(id)
    env = gym.make(id)
    env = env.unwrapped
    bounds, state_size = getBounds_Statesize(id)
    if RECORD:
        env = Monitor(env,'./cartpole-experiment-0201', force=True)
        observation = env.reset()
        state = observation2state(observation, state_size, bounds)
        for j in range(20000):
            #env.render()
            action = np.argmax(q[state])
            observation, reward, done, info = env.step(action)
            new_state=observation2state(observation,state_size,bounds)
            state=new_state
            if done:
                print(j)
                break
        env.close()
    episode=200
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
            if done or j==train_size-1:
                result.append(j+1)
                break
    result=np.array(result)
    if id!='CartPole-v0':
        result=-result
    plt.plot(result)
    plt.xlabel("number")
    plt.ylabel("reward")
    plt.show()
    print("mean",np.mean(result))
    print("var",np.std(result))
    print("len",len(result))

def getBounds_Statesize(id):
    env=gym.make(id)
    bounds=list(zip(env.observation_space.low,env.observation_space.high))
    #print(bounds)
    if id=='CartPole-v0':
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
    if __name__ == '__main__':
        num = input("Please input task number:\n 1.CartPole-v0 2.MountainCar-v0 3.Acrobot-v1 \n")
        if num == "1":
            test('CartPole-v0')
        elif num == "2":
            test('MountainCar-v0')
        elif num == "3":
            test('Acrobot-v1')
        else:
            print("wrong choice")
    #test('MountainCar-v0')
    #test('Acrobot-v1')