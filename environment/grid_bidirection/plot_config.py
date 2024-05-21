import enum
import gym
import gym_minigrid
import numpy as np
import matplotlib.pyplot as plt
import yaml

from simple_grid import MultiRoomGrid




# import wrappers

def read_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    # Read the ENV_CONFIG from the YAML file
    config_path = 'config/multi_grid_bidirection.yaml'
    config = read_config_from_yaml(config_path)
    ENV_CONFIG = config['ENV_CONFIG']

    # Create an environment instance with your configuration
    env = MultiRoomGrid(
        config=ENV_CONFIG["config"],
        start_rooms=ENV_CONFIG["start_rooms"],
        goal_rooms=ENV_CONFIG["goal_rooms"],
        room_size=ENV_CONFIG["room_size"],
        max_steps=ENV_CONFIG["max_steps"],
        num_rubble=ENV_CONFIG["num_rubble"],
        rubble_reward=1
    )

    # env = wrappers.FullyObsWrapper(env)
    # env = wrappers.ActionMasking(env)
    # env = wrappers.KeyPickupBonus(env)
    # env = wrappers.EncodeDirection(env)

    # Reset the environment (this will call _gen_grid and generate the grid)
    obs = env.reset()

    # Render the environment
    img = env.render('rgb_array')

    # print ('obs', obs)
    # m, n = len(obs['image']), len(obs['image'][0])
    # for r in range(m):
    #     for c in range(n):
    #         if obs['image'][r][c][0] == 10:
    #             print (obs['image'][r][c]) 

    # Save the image
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("grid.png", bbox_inches='tight', pad_inches=0)
    plt.show()