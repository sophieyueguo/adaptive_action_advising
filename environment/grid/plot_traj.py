import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Path to the .npy file
path = "states-usar.npy"

def extract_agent_positions_from_npy(file_path):
    # Load data from the npy file
    positions = np.load(file_path, allow_pickle=True)
    return positions

# Extract the agent positions from the .npy file
positions = extract_agent_positions_from_npy(path)

# Assuming the positions are directly the observations you mentioned earlier
observations = positions


def normalize_and_scale_rgb(raw_data):
    """
    Normalize and scale RGB values in the data.
    """
    m, n = len(raw_data), len(raw_data[0])
    data = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            value = round(raw_data[i][j][0])
            if value == 1:
                data[i][j] = [255, 255, 255]  # White
            elif value == 2:
                data[i][j] = [128, 128, 128]  # Grey
            elif value == 3:
                data[i][j] = [255, 255, 255]  # room entrance, same as empty space Red
            elif value == 5:
                data[i][j] = [0, 0, 255] # key
            elif value == 8:
                data[i][j] = [0, 255, 0]  # Green
            elif value == 10:
                data[i][j] = [255, 0, 0]  # red
            else:
                data[i][j] = [0, 0, 0]  # unclassified
            
    return np.array(data, dtype=np.uint8)


def process(observations, i):
    return np.flipud(normalize_and_scale_rgb(np.rot90(observations[i][0]['image'].astype(np.uint8))))


# Setup for the animation
fig, ax = plt.subplots()
im = ax.imshow(process(observations, 0), cmap='spring')
plt.axis('off')

def update(frame):
    """Update the image for a new frame."""
    rotated_image = np.flipud(normalize_and_scale_rgb(np.rot90(observations[frame][0]['image'].astype(np.uint8))))
    im.set_array(rotated_image)
    return [im]
    

ani = FuncAnimation(fig, update, frames=len(observations), blit=True, repeat=False)

# To show the animation inline (e.g., in a Jupyter notebook or IDE with plotting support)
plt.show()

# Uncomment the line below if you want to save the animation to a file
# ani.save('traj_animation.mp4', writer='ffmpeg', fps=30)
ani.save('traj_animation.gif', writer='pillow', fps=30)
