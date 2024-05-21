import matplotlib.pyplot as plt
import re

def read_data_and_plot(file_path):
    # Initialize lists to store epochs, training losses, and test losses
    epochs, train_losses, test_losses = [], [], []

    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Extract training data
            train_match = re.match(r'Epoch \[(\d+)/\d+\], Train Loss: ([\d\.]+)', line)
            if train_match:
                epoch, train_loss = train_match.groups()
                epochs.append(int(epoch))
                train_losses.append(float(train_loss))
            
            # Extract testing data
            test_match = re.match(r'Epoch \[(\d+)/\d+\], Test Loss: ([\d\.]+)', line)
            if test_match:
                test_losses.append(float(test_match.group(2)))

    # Plotting the data
    plt.figure(figsize=(12, 6))
    # plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss per Training Epoch')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig('/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/training_test_loss_plot.png')

# Usage
read_data_and_plot('/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/error_loss.txt')
