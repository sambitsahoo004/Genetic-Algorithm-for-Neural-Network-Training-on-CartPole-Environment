import gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Load the trained agent (change this path to your trained model)
model_path = '/home/sambit/Downloads/ga.zip'  # Replace with the full path to your trained model
model = PPO.load(model_path)

# Wrap the environment in a vectorized environment
env = DummyVecEnv([lambda: env])

# Number of episodes to visualize
num_episodes = 10

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
env.close()

