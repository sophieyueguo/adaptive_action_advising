import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pickle

# Load the trajectory
# Path to your text file
file_path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/data.txt'

# Initialize an empty list to hold the grid texts
grids = []

# Open and read the file
with open(file_path, 'r') as file:
    content = file.read()

# Split the content by double newlines (assuming this pattern is consistent as a separator)
parts = content.strip().split('\n\n')

for part in parts:
    lines = part.strip().split('\n')
    grid_lines = []
    capture = False  # Flag to start capturing lines for a grid
    for line in lines:
        if line.startswith('%'):
            capture = True  # Start capturing lines when '%' is encountered
        if capture:
            grid_lines.append(line)
        if line.startswith('Score:'):
            break  # Stop capturing lines when 'Score:' is encountered

    # Join the grid lines back into a single string and add to the grids list
    if grid_lines:
        grid_text = '\n'.join(grid_lines)
        grids.append(grid_text)

# Now 'grids' contains only the grid parts of the text
for grid in grids:
    print(grid)
    print()  # Print a newline for better separation in output



# Assuming all frames in the trajectory have the same size
# Calculate width and height from the first frame
first_frame_lines = grids[0].strip().split('\n')
height = len(first_frame_lines) - 1  # Exclude the score line
width = len(first_frame_lines[0])

fig, ax = plt.subplots(figsize=(width / 2, height / 2))
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor('black')

ghost_colors = ['pink', 'pink', 'pink', 'pink']  # Different colors for ghosts

# Function to draw a single frame
def draw_frame(frame_number):
    ax.clear()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('black')

    ghost_count = 0  # Reset ghost count for each frame

    grid = grids[frame_number]
    grid_lines = grid.strip().split('\n')
    score_line = grid_lines[-1]
    grid_lines = grid_lines[:-1]
    score_text = score_line.strip()



    for y, line in enumerate(grid_lines):
        for x, char in enumerate(line):
            if char == '%':
                # Draw walls
                square = plt.Rectangle((x, height - y - 1), 1, 1, color='grey')
                ax.add_patch(square)
            elif char == '.':
                # Draw small pellets
                circle = plt.Circle((x + 0.5, height - y - 0.5), 0.2, color='green')
                ax.add_patch(circle)
            elif char == 'o':
                # Draw power pellets
                circle = plt.Circle((x + 0.5, height - y - 0.5), 0.3, color='blue')
                ax.add_patch(circle)

            elif char in ['>', '<', '^', 'v']:
                # Draw Pacman with direction
                if char == '>':
                    start_angle, end_angle = 210, 150 
                elif char == '<':
                    start_angle, end_angle = 30, 330
                elif char == '^':
                    start_angle, end_angle = 300, 240
                elif char == 'v':
                    start_angle, end_angle = 120, 60
                
                pacman = patches.Wedge((x + 0.5, height - y - 0.5), 0.5, start_angle, end_angle, color='orange')
                ax.add_patch(pacman)

            elif char == 'G':
                # Draw ghosts with different colors and simple ghost shape
                ghost_color = ghost_colors[ghost_count % len(ghost_colors)]
                ghost_count += 1
                
                # Body
                ghost_body = patches.Circle((x + 0.5, height - y - 0.5), 0.5, color=ghost_color)
                ax.add_patch(ghost_body)
                
                # Eyes
                eye_left = patches.Circle((x + 0.3, height - y - 0.3), 0.1, color='white')
                ax.add_patch(eye_left)
                eye_right = patches.Circle((x + 0.7, height - y - 0.3), 0.1, color='white')
                ax.add_patch(eye_right)
                
                # Eye pupils
                pupil_left = patches.Circle((x + 0.3, height - y - 0.3), 0.05, color='black')
                ax.add_patch(pupil_left)
                pupil_right = patches.Circle((x + 0.7, height - y - 0.3), 0.05, color='black')
                ax.add_patch(pupil_right)


    # Display the score at the bottom
    plt.text(width / 2, 0.5, score_text, color='white', ha='center', va='center')

# Create the animation
anim = FuncAnimation(fig, draw_frame, frames=len(grids), interval=200)

# Save the animation
anim.save('pacman-tmp.mp4', writer='ffmpeg', fps=5, extra_args=['-vcodec', 'mpeg4'])


# Close the plot
plt.close(fig)
