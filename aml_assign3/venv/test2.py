'''import gym
env = gym.make('Acrobot-v1')
print(list(zip(env.observation_space.low,env.observation_space.high)))
print(env.observation_space.low)
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(action)
        #print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
'''
from matplotlib import pyplot as plt
import math
episode=2000
e=[]
alpha=[]
for i in range(episode):
    e.append(max(0.01, 2 - 2 / (1 + math.exp((-i / 30)))))
    # e=max(0.01,(1-i/episode))
    alpha.append(max(0.1, min(0.5, 1 / (1 + math.exp(((i - 30) / 80))))))

plt.figure(1)
plt.plot(range(episode),e,label="$epsilon$",c='r')
plt.plot(range(episode),alpha,label="$alpha$",c='b')
plt.legend()
plt.show()