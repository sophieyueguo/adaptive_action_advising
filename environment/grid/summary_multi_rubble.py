import os
import numpy as np

def process_file(file_path):
    # Initialize variables
    total_ball_count = 0
    line_count = 0
    line_with_500_count = 0

    # Reading the file and processing the contents
    try:
        with open(file_path, 'r') as file:
            for i, line in enumerate(file, 1):
                if line.strip() == '500':
                    line_with_500_count += 1
                elif 'ball count' in line:
                    try:
                        count = int(line.strip().split(' ')[-1])
                        total_ball_count += count
                        line_count += 1
                    except ValueError:
                        # Handle potential conversion error
                        pass

        # Calculating the average
        average_ball_count = total_ball_count / line_count if line_count else 0

        return line_with_500_count, average_ball_count

    except FileNotFoundError:
        print("File not found. Please ensure the file path is correct.")
        return None, None



def process_all_files(root_path, summary_file_name):
    # Find all txt files in the directory
    txt_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_path) for f in filenames if os.path.splitext(f)[1] == '.txt']

    # Open the summary file
    with open(summary_file_name, 'w') as summary_file:
        for file_path in txt_files:
            # Process each file and get the summary
            print ('file_path', file_path)
            line_with_500_count, average_ball_count = process_file(file_path)
            if line_with_500_count is not None:
                # Write the summary to the summary file
                summary_file.write(f"File: {file_path}\n")
                summary_file.write(f"The victim count not found is {line_with_500_count}.\n")
                summary_file.write(f"The average ball count is {average_ball_count}.\n\n")
            else:
                summary_file.write(f"Unable to process the file: {file_path}\n\n")
            

# Usage
# root_path = '/home/glow/ray_results/AdvisedTrainer_2024-01-19_20-28-05'  # Replace with the path where your txt files are located
#root_path = '/home/glow/ray_results/AdvisedTrainer_2024-01-20_01-20-46'
root_path = '/home/glow/ray_results/AdvisedTrainer_2024-01-20_12-34-44'
summary_file_name = root_path + '/summary.txt'
process_all_files(root_path, summary_file_name)
