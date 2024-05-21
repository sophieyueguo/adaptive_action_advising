import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

from world_model import train_rewardNN
#import train_rewardNN

def exp_finetune_nn(data_type):
    X = np.load('/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/student_X.npy', allow_pickle=True)
    Y = np.load('/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/student_Y.npy', allow_pickle=True)
    print ('X_shape', X.shape)
    print('X[0][0].shape', X[0][0].shape)
    print ('X[0][1].shape', X[0][1])
    print ('Y_shape', Y.shape)
    train_dataset, test_dataset = train_rewardNN.process_data(X, Y, data_type=data_type, normalization=1)
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=True)


    loaded_model_path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/rewardNN_500iter.pth'
    model = train_rewardNN.CNNRewardModel()
    model.load_state_dict(torch.load(loaded_model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = train_rewardNN.WeightedMSELoss(10)
    model.train()
    model = train_rewardNN.train_model('rewardNN', train_loader, 
                        criterion, 
                        model=model, save_model=True, model_path=None, 
                        train_iter=0, lr=0.00001)
    error_counter = train_rewardNN.test_rewardNN(model, criterion, train_loader, None)
    print ('train error counter', error_counter)
    error_counter = train_rewardNN.test_rewardNN(model, criterion, test_loader, None)
    print ('test error counter', error_counter)



def save_teacher_errors():
    path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/'

    X = np.load(path + 'student_X.npy', allow_pickle=True)
    Y = np.load(path + 'student_Y.npy', allow_pickle=True)
    
    print ('X_shape', X.shape)
    print('X[0][0].shape', X[0][0].shape)
    print ('X[0][1].shape', X[0][1])
    print ('Y_shape', Y.shape)

    normalization=1

    loaded_model_path = path + 'rewardNN_wgt10-15.pth'
    model = train_rewardNN.CNNRewardModel()
    model.load_state_dict(torch.load(loaded_model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    tp, fp, tn, fn = 0, 0, 0, 0
    errors = []

    for i in range(len(X)):
        obs_, action_ = X[i]
        
        tensor_obs = torch.tensor(obs_.transpose(2, 0, 1) / normalization, dtype=torch.float32).unsqueeze(0)
        tensor_action = torch.tensor([action_ /normalization], dtype=torch.long)
        tensor_obs, tensor_action = tensor_obs.to(device), tensor_action.to(device)
        
        predicted_reward = model(tensor_obs, tensor_action).item()

        if Y[i] > 0:
            if predicted_reward > 0.5:
                tp += 1
            else:
                fn += 1
        else:
            if predicted_reward > 0.5:
                fp += 1
            else:
                tn += 1
        errors.append(predicted_reward - Y[i])

    print ('tp, fp, tn, fn', tp, fp, tn, fn)
    print ('len of errors', len(errors))
    np.save(path + 'student_errors.npy', errors)
    

def exp_exmaine_error_correction(data_type):
    path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/'

    reward_model_path = path + 'rewardNN_wgt10-15.pth' 
    error_model_path = path + 'error_correction.pth'
    
    # Load data
    X = np.load(path + 'student_X.npy', allow_pickle=True)
    Y = np.load(path + 'student_Y.npy', allow_pickle=True)

    train_indices = np.load(path + 'train_indices.npy')
    test_indices = np.load(path + 'test_indices.npy')

    X = X[test_indices]
    Y = Y[test_indices]


    # Load the reward prediction model
    reward_model = train_rewardNN.CNNRewardModel()
    reward_model.load_state_dict(torch.load(reward_model_path))
    reward_model.to(train_rewardNN.device)
    reward_model.eval()

    # Load the error prediction model
    error_model = TransformerModel()  # Ensure this is the same architecture as the trained one
    error_model.load_state_dict(torch.load(error_model_path))
    error_model.to(train_rewardNN.device)
    error_model.eval()

    # Iterate through data and correct rewards
    normalization = 1
    tp, fp, tn, fn = 0, 0, 0, 0

    fp_obs_action = []
    fn_obs_action = []
    for i in range(len(X)):
        obs_, action_ = X[i]
        
        tensor_obs = torch.tensor(obs_.transpose(2, 0, 1) / normalization, dtype=torch.float32).unsqueeze(0)
        tensor_action = torch.tensor([action_ / normalization], dtype=torch.long)
        tensor_obs, tensor_action = tensor_obs.to(train_rewardNN.device), tensor_action.to(train_rewardNN.device)
        
        with torch.no_grad():
            predicted_reward = reward_model(tensor_obs, tensor_action).item()
            predicted_error = error_model(tensor_obs, tensor_action).item()
            # print ('predicted_reward', predicted_reward, 'predicted_error', predicted_error)

        # Adjust reward prediction by adding predicted error
        adjusted_reward = predicted_reward - predicted_error
        # adjusted_reward = predicted_reward

        if Y[i] > 0:
            if adjusted_reward > 0.5:
                tp += 1
            else:
                fn += 1
                fn_obs_action.append([obs_, action_])
        else:
            if adjusted_reward > 0.5:
                fp += 1
                fp_obs_action.append([obs_, action_])
            else:
                tn += 1

    print('tp, fp, tn, fn', tp, fp, tn, fn)
    np.save('world_model/debug_fp/fp_pairs.npy', fp_obs_action)
    np.save('world_model/debug_fn/fn_pairs.npy', fn_obs_action)   


def load_saved_data(path):
    
    X = np.load(path + 'student_X.npy', allow_pickle=True)
    Y = np.load(path + 'student_errors.npy', allow_pickle=True)
    print('student X.shape', X.shape, 'student Y.shape', Y.shape)
    normalization = 1
    
    # train_dataset, test_dataset = process_data(X, Y, data_type=data_type, normalization=1)
    X = [(torch.tensor(obs.transpose(2, 0, 1)/normalization, dtype=torch.float32),
             torch.tensor(action/normalization, dtype=torch.long))
              for obs, action in X]
    
    Y = torch.tensor(Y, dtype=torch.float32)
    return X, Y

    
def exp_train_error_model(X=None, Y=None, model=None, save_path=None):
    path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/'
    if X == None and Y == None:
        X, Y = load_saved_data(path)
    print('X[0].shape', X[0][0].shape, 'X[0][1]', X[0][1])
    print('Y[0]', Y[0])
    
    dataset = train_rewardNN.Dataset(X, Y)

    # Define the size of the splits
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Use a fixed generator for reproducibility
    generator = torch.Generator().manual_seed(42)

    # Get indices for the entire dataset
    indices = torch.randperm(len(dataset), generator=generator).tolist()

    # Split indices for training and testing
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create subsets using indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Save indices if needed
    # np.save(path + 'train_indices.npy', train_indices)
    # np.save(path + 'test_indices.npy', test_indices)

    print ('len(train_indices)', len(train_indices), 
           'len(test_indices)', len(test_indices))

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

    num_epochs = 100#100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model == None:
        model = TransformerModel()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # criterion = nn.MSELoss()
    criterion = train_rewardNN.WeightedMSELoss(10) #100

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for (obs, action), error in train_loader:
            obs, action, error = obs.to(device), action.to(device), error.to(device)

            optimizer.zero_grad()
            predictions = model(obs, action)
            loss = criterion(predictions.squeeze(), error)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        # Evaluate on the test set
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for (obs, action), error in test_loader:
                obs, action, error = obs.to(device), action.to(device), error.to(device)
                predictions = model(obs, action)
                loss = criterion(predictions.squeeze(), error)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}")

    if save_path != None:
        torch.save(model.state_dict(), save_path)
    return model








class TransformerModel(nn.Module):
    def __init__(self, img_embedding_size=256, action_embedding_size=10, num_heads=2, num_layers=3, num_classes=1):
        super(TransformerModel, self).__init__()
        # Convolutional layers for image encoding
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Image flattening
        self.img_flatten_size = 268800  # Adjusted according to new conv layers output
        self.img_fc = nn.Linear(self.img_flatten_size, img_embedding_size)
        
        # Action embedding
        self.action_embed = nn.Embedding(10, action_embedding_size)  # Assuming 10 possible actions

        # Transformer Encoder
        # Calculate total embedding size and print to debug
        d_model = img_embedding_size + action_embedding_size
        print(f"Image Embedding Size: {img_embedding_size}")
        print(f"Action Embedding Size: {action_embedding_size}")
        print(f"Total Embedding Size (d_model): {d_model}")
        print(f"Number of Heads (num_heads): {num_heads}")

        assert d_model % num_heads == 0, "Total embedding size must be divisible by number of heads"

        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, img, action):
        # Image processing
        x = F.relu(self.conv1(img))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.img_flatten_size)  # Flatten the image with updated size
        img_embedded = F.relu(self.img_fc(x))
        
        # Action embedding
        action_embedded = self.action_embed(action).squeeze(1)
        
        # Combine image and action embeddings
        combined = torch.cat((img_embedded, action_embedded), dim=1)
        
        # Transformer Encoder
        transformer_output = self.transformer_encoder(combined.unsqueeze(1))
        
        # Output
        error_prediction = self.fc_out(transformer_output.squeeze(1))
        return error_prediction




#save_teacher_errors()


# data_type = 'rewardNN'
# exp_exmaine_error_correction(data_type)


# data_type = 'rewardError'
# save_path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/error_correction.pth'
# exp_train_error_model(save_path=save_path)


#exp_finetune_nn(data_type)