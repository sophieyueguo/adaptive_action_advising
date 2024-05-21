#!/bin/bash

# =========================================
# make sure the config is "never advise"...
# =========================================

# Set your root folder path here
# no advice r=1
#root_path="/home/glow/ray_results/AdvisedTrainer_2024-01-18_11-30-48"
#root_path="/home/glow/ray_results/AdvisedTrainer_2024-01-19_20-28-05"
#root_path="/home/glow/ray_results/AdvisedTrainer_2024-01-20_01-20-46"
root_path="/home/glow/ray_results/AdvisedTrainer_2024-01-20_12-34-44"


# Extract the model configuration from the directory structure or another source
model_config="multi_grid_14room_10rubble"

# Loop through each progress.csv file
find "$root_path" -name 'progress.csv' | while read -r progress_file; do
    model_dir=$(dirname "$progress_file")

    # Find the latest checkpoint file in the model directory
    latest_checkpoint=$(ls -t "$model_dir"/checkpoint_* | head -n 1)

    # Correctly format the latest checkpoint path to remove any trailing colon
    latest_checkpoint=$(echo "$latest_checkpoint" | sed 's/[:]$//')

    # Extract the checkpoint number from the latest checkpoint path
    checkpoint_number=$(basename "$latest_checkpoint" | sed 's/checkpoint_//')

    # Extract the parent directory name to use in the export path
    parent_dir_name=$(basename "$model_dir")

    # Construct the export path including the checkpoint number
    export_path="${root_path}/${parent_dir_name}_${checkpoint_number}"

    # Echo the export command for verification
    echo "Executing export: python train.py --mode=export --config=$model_config --checkpoint-dir=$latest_checkpoint --export-path=$export_path"

    # Execute the export command
    python train.py --mode=export --config=$model_config --checkpoint-dir="$latest_checkpoint" --export-path="$export_path"

    # Construct the evaluation command
    eval_command="python train.py --mode=evaluate --config=$model_config --import-path=$export_path --eval-episodes=1000"

    # Construct the output file path for evaluation results
    eval_output="${export_path}.txt"

    # Echo the evaluation command for verification
    echo "Executing evaluate: $eval_command > $eval_output"

    # Execute the evaluation command and redirect output to a text file
    eval $eval_command > "$eval_output"
done

