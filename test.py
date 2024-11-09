import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('CartPole-v1')
print(env.reset())
print(env.step(0))


