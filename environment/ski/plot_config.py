import enum
import gym

import numpy as np
import matplotlib.pyplot as plt
import yaml

import random


# import wrappers

def read_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    # Read the ENV_CONFIG from the YAML file
    config_path = 'config/Skiing-v4_stacked.yaml'
    config = read_config_from_yaml(config_path)
    ENV_CONFIG = config['ENV_CONFIG']

    # Create an environment instance with your configuration
    env = gym.make("Skiing-v4")
    # env = wrappers.ObsWrapper(env)
    obs = env.reset()

    done = False
    sum_reward = 0
    t = 0
    while not done:
        #action = random.choice([0, 1, 2])
        action = 0
        obs, reward, done, info = env.step(action)
        print(t, reward, info, obs.shape)
        sum_reward += reward
        t += 1
    print('sum_reward', sum_reward)

    env_attributes = dir(env)
    print(env_attributes)


    original_env = env.__getattr__
    print(dir(original_env))
    
        
        

    # Render the environment
    # img = env.render('rgb_array')
    
    #TODO feature bigger range

    # Save the image
    # plt.imshow(img)
    
    plt.imshow(obs)
    plt.axis('off')
    plt.savefig("grid.png", bbox_inches='tight', pad_inches=0)
    plt.show()