import enum
import gym
import gym_minigrid
import numpy as np
import matplotlib.pyplot as plt
import yaml
import random

import config.config_loader
import environment.env_loader
import model.model_wrapper
import model.util
import pickle
import ray
import teacher_student.util
import tempfile
import torch
from datetime import datetime

from ray import tune
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.logger import pretty_print, UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR


from ray.rllib.utils.torch_utils import convert_to_torch_tensor





def read_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def default_logger(config):
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}_{}".format(str(config["alg"]), config["env"].replace("/", "_"), timestr)

    logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=DEFAULT_RESULTS_DIR)

    def default_logger_creator(config):
        """Creates a Unified logger with the default prefix."""
        return UnifiedLogger(config, logdir, loggers=None)

    return default_logger_creator

if __name__ == "__main__":

    ray.init(local_mode = False)

    model.util.register_models()
    environment.env_loader.register_envs()

    exp_config = config.config_loader.ConfigLoader.load_config('multi_grid_14room')

    model_type = model.model_wrapper.ModelType.TORCH

    exp_config = teacher_student.util.get_trainer(exp_config["alg"])(config=exp_config, logger_creator=default_logger(exp_config)).config
    env = environment.env_loader.env_maker(exp_config["env_config"], exp_config["env"])

    loaded_model = model.model_wrapper.ModelWrapper(model_type)
    import_path = 'test_model'
    loaded_model.load(import_path, env.action_space, env.observation_space, exp_config["model"])

    


    # # Read the ENV_CONFIG from the YAML file
    # config_path = 'config/multi_grid_14room.yaml'
    # config = read_config_from_yaml(config_path)
    # ENV_CONFIG = config['ENV_CONFIG']

    # # Create an environment instance with your configuration
    # env = MultiRoomGrid(
    #     config=ENV_CONFIG["config"],
    #     start_rooms=ENV_CONFIG["start_rooms"],
    #     goal_rooms=ENV_CONFIG["goal_rooms"],
    #     room_size=ENV_CONFIG["room_size"],
    #     max_steps=ENV_CONFIG["max_steps"]
    # )

    # env = wrappers.FullyObsWrapper(env)
    # env = wrappers.ActionMasking(env)
    # env = wrappers.PickupBonus(env)
    

    # env = wrappers.KeyPickupBonus(env)
    # env = wrappers.EncodeDirection(env)

    # Reset the environment (this will call _gen_grid and generate the grid)
    obs = env.reset()

    # Render the environment
    img = env.render('rgb_array')


    done = False
    sum_reward = 0
    t = 0
    frames = []  # List to store frames

    while not done:
        # action = random.choice([0, 1, 2, 3, 4, 5, 6])
        #action = 0
        # action_mask = obs['action_mask']

        # allowed_indices = [index for index, value in enumerate(action_mask) if value == 1]

        # # Randomly choose one index from the allowed indices
        # action = np.random.choice(allowed_indices, size=1)[0]

        action, _, importance = loaded_model.get_action(obs)

        obs, reward, done, info = env.step(action)
        print('t, reward', t, reward)
        img = env.render('rgb_array')
        sum_reward += reward
        t += 1

        # Add the current observation to frames
        frames.append(img)

    print('sum_reward', sum_reward)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

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
    ani.save('animation.gif', writer='pillow', fps=10)