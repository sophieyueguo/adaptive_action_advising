import matplotlib.pyplot as plt
import matplotlib.patches as patches  # Import the patches module
import numpy as np

# Define your grid here
grid = """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%............%%............%
%.%%%%.%%%%%.%%.%%%%%.%%%%.%
%o%%%%.%%%%%.%%.%%%%%.%%%%o%
%.%%%%.%%%%%.%%.%%%%%.%%%%.%
%..........................%
%.%%%%.%%.%%%%%%%%.%%.%%%%.%
%.%%%%.%%.%%%%%%%%.%%.%%%%.%
%......%%..G.%%....%%......%
%%%%%%.%%%%% %%G%%%%%.%%%%%%
%%%%%%.%%%%% %% %%%%%.%%%%%%
%%%%%%.%            %.%%%%%%
%%%%%%.% %%%%  %%%% %.%%%%%%
%     .  %G  G    %  .     %
%%%%%%.% %%%%%%%%%% %.%%%%%%
%%%%%%.%            %.%%%%%%
%%%%%%.% %%%%%%%%%% %.%%%%%%
%............%%............%
%.%%%%.%%%%%.%%.%%%%%.%%%%.%
%.%%%%.%%%%%.%%.%%%%%.%%%%.%
%o..%%.......  .......%%..o%
%%%.%%.%%.%%%%%%%%.%%.%%.%%%
%%%.%%.%%.%%%%%%%%.%%.%%.%%%
%......%%....%% ...%%......%
%.%%%%%%%%%%.%% %%%%%%%%%%.%
%...........    > .........%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Score: -396
"""

grid_lines = grid.strip().split('\n')
score_line = grid_lines[-1]  # Last line contains the score
grid_lines = grid_lines[:-1]  # All lines except the last one
score_text = score_line.strip()  # Extract the score text
height = len(grid_lines)
width = len(grid_lines[0])

fig, ax = plt.subplots(figsize=(width / 2, height / 2))
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_aspect('equal')
ax.axis('off')  # Turn off the axis

ax.set_facecolor('black')  # Set background to black

ghost_colors = ['pink', 'pink', 'pink', 'pink']  # Different colors for ghosts
ghost_count = 0

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
                start_angle, end_angle = 30, 330
            elif char == '<':
                start_angle, end_angle = 210, 150
            elif char == '^':
                start_angle, end_angle = 120, 60
            elif char == 'v':
                start_angle, end_angle = 300, 240
            
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
plt.show()



