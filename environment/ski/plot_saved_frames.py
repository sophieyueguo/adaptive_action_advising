import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yaml
import random
import wrappers

def read_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Assuming 'frames' is your list of observations
    frames = np.load('states.npy', allow_pickle=True)
    observations = frames
    

    def process(observation):
        """
        Process the observation for visualization.
        """
        # Assuming observation is already a NumPy array in the correct format
        # Add any specific processing if needed
        # print(observation)
        observation = observation[0].astype(np.float32)
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
