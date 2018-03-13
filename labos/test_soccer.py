import gym_soccer
import gym

env = gym.make('Soccer-v0')
env.reset()
env.render()

print(env.actions)

#for _ in range(1000):
#    env.render()

