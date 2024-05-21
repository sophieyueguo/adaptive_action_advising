
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yaml
import random
import wrappers

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation





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
    env = wrappers.ExtractFeature(env)
    env = wrappers.ObsWrapper(env)
    obs = env.reset()

    done = False
    sum_reward = 0
    t = 0
    frames = []  # List to store frames
    
    
    while not done:
    #while t < 500:
        action = random.choice([0, 1, 2])
        #action = 0
        obs, reward, done, info = env.step(action)

        sum_reward += reward
        t += 1

        # Add the current observation to frames
        frames.append(obs)

    print('sum_reward', sum_reward)











    # Assuming 'frames' is your list of observations
    observations = frames

    def process(observation):
        """
        Process the observation for visualization.
        """
        # Assuming observation is already a NumPy array in the correct format
        # Add any specific processing if needed
        return observation

    # Setup for the animation
    fig, ax = plt.subplots()
    im = ax.imshow(process(observations[0]), cmap='spring')
    plt.axis('off')

    def update(frame):
        """Update the image for a new frame."""
        processed_image = process(observations[frame])
        im.set_array(processed_image)
        return [im]

    ani = FuncAnimation(fig, update, frames=len(observations), blit=True, repeat=False)

    # To show the animation inline (e.g., in a Jupyter notebook or IDE with plotting support)
    plt.show()

    # Uncomment the line below if you want to save the animation to a file
    # ani.save('animation.mp4', writer='ffmpeg', fps=30)
    ani.save('animation.gif', writer='pillow', fps=30)
