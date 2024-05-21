import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class CNNRewardModel(nn.Module):
    def __init__(self, action_embed_size=32):
        super(CNNRewardModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)

        # Attention layer
        self.attention = nn.Linear(64, 1)

        # Embedding layer for action
        self.action_embed = nn.Embedding(7, action_embed_size)

        # Interaction layer for feature combination
        # Adjust the input dimension based on your feature map size after conv layers
        feature_map_height, feature_map_width = 1, 1
        conv_output_size = 64 * feature_map_height * feature_map_width
        self.interaction_layer = nn.Linear(conv_output_size + action_embed_size, 64)
        

        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, image, action):
        x = F.relu(self.conv1(image))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Reshape for attention
        batch_size, num_channels, height, width = x.shape
        x = x.view(batch_size, num_channels, -1)
        x = x.permute(0, 2, 1)

        # Compute attention weights for each spatial location
        attention_weights = F.softmax(self.attention(x), dim=2)

        # Apply attention weights
        x = x * attention_weights

        # Aggregate results
        x = x.sum(dim=1)
        x = x.view(batch_size, -1)

        # Embed the action
        action_emb = F.relu(self.action_embed(action))
        action_emb = action_emb.view(action_emb.size(0), -1)

        # Combine action embeddings with image features
        combined = torch.cat([x, action_emb], dim=1)

        # Feature interaction
        x = F.relu(self.interaction_layer(combined))

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x






def process_data(X, Y, data_type='rewardNN', normalization=1):
    X = [(torch.tensor(obs.transpose(2, 0, 1)/normalization, dtype=torch.float32),
             torch.tensor(action/normalization, dtype=torch.long))
              for obs, action in X]
    
    if data_type == 'rewardNN':
        # for usar we put a binary
        Y[Y > 0] = 1 
        Y = torch.tensor(Y, dtype=torch.float32)
    elif data_type == 'rewardError':
        Y = torch.tensor(Y, dtype=torch.float32)
    else:
        print ('unknown name')
        exit()

    # Split the dataset
    train_size = int(0.8 * len(Y))
    test_size = len(Y) - train_size
    train_dataset, test_dataset = random_split(Dataset(X, Y), [train_size, test_size])

    print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print ('data sized loaded:')
    print (len(X))
    print (len(Y))
    return train_dataset, test_dataset



def load_data_from_batch(batches, data_type='rewardNN', normalization=1):
    X = []
    Y = []

    for batch in batches:
        batch_size = batch['rewards'].numel()
        for i in range(batch_size): 

            # if batch['rewards'][i] > 0 and batch['rewards'][i] < 0.6:
            #     print ('rewards', batch['rewards'][i])
            #     print ('output', batch['obs'][i][7:])
            #     print ('action', batch['actions'][i])
        
            o = batch['obs'][i][7:].cpu().numpy().reshape(17, 17, 3)
            r = batch['rewards'][i].item()
            a = batch['actions'][i].item()
            X.append((o, a)) 
            Y.append(r)

            # if r == 0.55:
            #     print ('reward of removing rubble recorded')

    # print ('X[0]', X[0])
    # print ('Y[0]', Y[0])
    Y = np.array(Y)
    # np.save('student_X', X)
    # np.save('student_Y', Y)
    # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!saved student data!!!!!!!!!!!!!!!!!!!!!!')    
    train_dataset, test_dataset = process_data(X, Y, data_type=data_type, normalization=normalization)

    # DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=65536, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=65536, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=True)

    return train_loader, test_loader, len(X)    






def load_data_from_file(data_path, data_type, normalization=1):
    # Assuming X, rewardNN_Y are already populated
    # Convert data to PyTorch tensors
    print ('data_path', data_path)
    X = np.load(data_path + 'X.npy', allow_pickle=True)
    X = [(obs['image'], action)
        for obs, action in X]
    Y = np.load(data_path + data_type + '_Y.npy', allow_pickle=True)
   
    train_dataset, test_dataset = process_data(X, Y, data_type=data_type, normalization=normalization)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=True)

    return train_loader, test_loader, len(X)


# Custom dataset
class Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X  
        self.Y = Y  

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        obs, action = self.X[idx]
        return (obs, action), self.Y[idx]
        

def train_model(data_type, train_loader, criterion, 
                model=None, save_model=False, model_path=None, train_iter=1000, lr=0.0001):
    # Initialize the model and move it to the appropriate device
    if data_type == 'rewardNN': 
        if model==None:
            model = CNNRewardModel().to(device)
    else:
        print ('unknown name')
        exit()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(train_iter): 
        model.train()
        for (obs, actions), y in train_loader:
            obs, actions, y = obs.to(device), actions.to(device), y.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(obs, actions)
            loss = criterion(outputs.squeeze(), y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
 
        print(f"Reward Training Epoch {epoch+1}, Loss: {loss.item()}")

    # Save the model if specified
    if save_model and model_path is not None:
        torch.save(model.state_dict(), model_path)

    return model


def test_rewardNN(model, criterion, test_loader, output_file_path):
    model.eval()
    error_counter = 0

    gt_one_count = 0

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    # with torch.no_grad(), open(output_file_path, 'w') as file:
    if True:
        test_loss = 0
        for (obs, actions), rewards in test_loader:
            obs, actions, rewards = obs.to(device), actions.to(device), rewards.to(device)
            outputs = model(obs, actions)

            # Write outputs and rewards to file
            for out, gt in zip(outputs, rewards):
                
                if gt.item() == 1:
                    gt_one_count += 1
                    # print ('out', out)

                out_rounded = out.round().item()  # Round to nearest integer
                gt_rounded = gt.round().item()    # Round to nearest integer
 
                if abs(gt_rounded-out_rounded) > 0:
                    # print(out, gt)
                    error_counter += 1
                    if gt_rounded == 0:
                        fp += 1
                    if gt_rounded == 1:
                        fn += 1
                else:
                    if gt_rounded == 0:
                        tn += 1
                    if gt_rounded == 1:
                        tp += 1
                # file.write(f"{out.item():.3f}, {gt.item():.3f}\n")

            test_loss += criterion(outputs.squeeze(), rewards).item()
        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss}")
    
    print ('gt_one_count', gt_one_count)
    print ('error_counter', error_counter)
    print ('tp, fp, tn, fn', tp, fp, tn, fn)
    return error_counter



class WeightedMSELoss(nn.Module):
    def __init__(self, weight_non_zero, weight_false_positive, weight_ones=0):
        super(WeightedMSELoss, self).__init__()
        self.weight_non_zero = weight_non_zero
        self.weight_false_positive = weight_false_positive
        self.weight_ones = weight_ones 

    def forward(self, input, target):
        # Create a weight tensor with higher values for non-zero targets
        weights = torch.ones_like(target)
        
        # Calculate MSE loss with weights
        loss = (input - target) ** 2
        
        rounded_input = torch.round(input)
        rounded_target = torch.round(target)

        addition_loss = (rounded_input - rounded_target) ** 2
        weights[addition_loss > 0] += self.weight_non_zero 

        false_positives = (rounded_input > rounded_target).float()  # True for false positives
        weights += false_positives * self.weight_false_positive

        weights[target > 0] += self.weight_ones
        
        loss = loss * weights
        return loss.mean()


def exp_train_nn(data_type, criterion):
    data_path = 'world_model/'
    train_loader, test_loader, length = load_data_from_file(data_path, data_type)
    model_path = data_path + data_type + '.pth'
    
    if data_type == 'rewardNN':
        model = train_model(data_type, train_loader, criterion, 
                            save_model = True, model_path=model_path, 
                            train_iter=1000, lr=0.00025)
                            # train_iter=500, lr=0.00075) 
                              
        error_counter = test_rewardNN(model, criterion, train_loader, data_path + 'reward_train_output.txt')
        print ('error_rate', error_counter/length)
        error_counter = test_rewardNN(model, criterion, test_loader, data_path + 'reward_test_output.txt')
        print ('error_rate', error_counter/length)


def exp_test_nn(data_type, criterion):
    data_path = 'world_model/'
    _, test_loader, length = load_data_from_file(data_path, data_type)
    model_path = data_path + data_type + '.pth'

    if data_type == 'rewardNN':
        model = CNNRewardModel()
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        error_counter = test_rewardNN(model, criterion, test_loader, data_path + 'reward_test_output.txt')
        print ('error_rate', error_counter/length)




data_type = 'rewardNN'

criterion = WeightedMSELoss(10, 15)
exp_train_nn(data_type, criterion)
#exp_test_nn(data_type, criterion)


