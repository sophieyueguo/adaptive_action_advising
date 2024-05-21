from gym.wrappers import FrameStack
from gym.core import ObservationWrapper
import numpy as np
import gym
import os.path as path
import os

# class StackFrameWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.last_obs = None

#     def reset(self, **kwargs):
#         obs, _ = self.env.reset(**kwargs)

#         return obs

#     def step(self, action):
#         observation, reward, terminated, truncated, info = self.env.step(action)

#         # Since RLlib doesn't support the truncated variable (yet), incorporate it into terminated
#         terminated = terminated or truncated
#         return observation, reward, terminated, info




from collections import deque

class ObsWrapper(ObservationWrapper):
    def __init__(self, env, frame_cnt=1):
        super().__init__(env)
        self.frame_cnt = frame_cnt
        self.frames = deque(maxlen=self.frame_cnt)
        self.obs_shape = (210, 160, 3)  # Adjust this if your observation shape is different
        self.unwrapped_obs = np.zeros(self.obs_shape)
    
    def reset(self, **kwargs):
        self.frames.clear()
        obs = self.env.reset(**kwargs)
        # Initialize with the first observation
        for _ in range(self.frame_cnt):
            self.frames.append(obs.astype(np.float32))
        return self.compute_mean_observation()

    def observation(self, obs):
        self.frames.append(obs.astype(np.float32))
        return self.compute_mean_observation()

    def compute_mean_observation(self):
        # Compute the mean across frames
        mean_obs = np.mean(self.frames, axis=0)

        # Apply ceiling to the floats and convert back to uint8
        mean_obs = np.ceil(mean_obs).astype(np.uint8)

        # mean_obs[mean_obs > 0] = 1
        self.unwrapped_obs = mean_obs
        return mean_obs



'''player color (214, 92, 92)
    gate color (66, 72, 200)
   final gate color (184, 50, 50)}'''
def find_top_gate_positions(image, y_min_contain_gate, x_min_contain_gate=25, x_max_contain_gate=145):
    # Exact colors for gates
    gate_color = np.array([66, 72, 200], dtype="uint8")
    final_gate_color = np.array([184, 50, 50], dtype="uint8")
    

    gate_positions = []
    final_gate_positions = []
    found_gate = False
    found_final_gate = False

    for y in range(y_min_contain_gate, image.shape[0]):  # Loop over rows (y coordinate)
        for x in range(x_min_contain_gate, x_max_contain_gate):  # Loop over columns (x coordinate)
            if np.array_equal(image[y, x], gate_color):
                gate_positions.append((x, y))
                found_gate = True
            elif np.array_equal(image[y, x], final_gate_color):
                final_gate_positions.append((x, y))
                found_final_gate = True
        

        # If a regular gate is found, fill in the line
        if found_gate:
            xs = [pos[0] for pos in gate_positions]
            if len(xs) > 1:
                for x_ in range(min(xs) + 1, max(xs)):
                    gate_positions.append((x_, y))
            break  # Stop after finding the first gate line

        # If the final gate is found, fill in the line
        if found_final_gate:
            xs = [pos[0] for pos in final_gate_positions]
            if len(xs) > 1:
                for x_ in range(min(xs) + 1, max(xs)):
                    final_gate_positions.append((x_, y))
            break  # Stop after finding the first final gate line

    return gate_positions, final_gate_positions, y










class RewardParaStudent(gym.Wrapper):
    """
    set up reward parameters.
    """

    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited (state,action) pairs.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        extra_reward = True
        if extra_reward:
            self.buggy_cumulative_reward = -100
            self.default_reward = -5
            
            self.pass_gate_reward = 100000
            self.fall_down_reward = -1000
            self.fall_down_between_gate_reward = 0
            self.step_reward = 0
            
            self.keep_info = False
        
        else:
            self.pass_gate_reward = 0
            self.fall_down_reward = 0
            self.fall_down_between_gate_reward = 0
            self.step_reward = 0
            self.buggy_cumulative_reward = -1000000000000
            self.default_reward = -1000000000000
            self.keep_info = True



class RewardParaTeacher(gym.Wrapper):
    """
    set up reward parameters.
    """

    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited (state,action) pairs.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        extra_reward = True
        if extra_reward:
            self.buggy_cumulative_reward = -100
            self.default_reward = -5

            self.pass_gate_reward = 10000
            self.fall_down_reward = -1000
            self.fall_down_between_gate_reward = 0
            self.step_reward = 0
            
            self.keep_info = False
        
        else:
            self.pass_gate_reward = 0
            self.fall_down_reward = 0
            self.fall_down_between_gate_reward = 0
            self.step_reward = 0
            self.buggy_cumulative_reward = -1000000000000
            self.default_reward = -1000000000000
            self.keep_info = True





class ExtractFeature(gym.Wrapper):
    """
    extract features of passing gates and falling down.
    """

    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited (state,action) pairs.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.gate_height = 10

        # extra_reward = True
        # finetuned_preference = False

        self.max_gate = 8#20

        # if extra_reward:
            
        #     self.buggy_cumulative_reward = -100
        #     self.default_reward = -5
        #     if not finetuned_preference:
        #         self.pass_gate_reward = 10000
        #         self.fall_down_reward = -1000
        #         self.fall_down_between_gate_reward = 0
        #         self.step_reward = 0
        #     else:
        #         self.pass_gate_reward = 100000
        #         self.fall_down_reward = -1000
        #         self.fall_down_between_gate_reward = 0
        #         self.step_reward = 0
        #     self.keep_info = False
        
        # else:
        #     self.pass_gate_reward = 0
        #     self.fall_down_reward = 0
        #     self.fall_down_between_gate_reward = 0
        #     self.step_reward = 0
        #     self.buggy_cumulative_reward = -1000000000000
        #     self.default_reward = -1000000000000
        #     self.keep_info = True

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, info = self.env.step(action)

        #override reward at the end - accumulated
        if reward < self.buggy_cumulative_reward:
            reward = self.default_reward
        
        reward += self.step_reward

        y_min_contain_gate = max([30, self.prev_gate_y - 10])
        gate_positions, final_gate_positions, gate_y = find_top_gate_positions(obs, y_min_contain_gate)
        

        top_line_contain_player = False
        bot_line_contain_player = False
        visualize_line = False

        if self.keep_info:
            info['fall down between gate'] = 0
            info['fall down near gate'] = 0
            info['pass gate'] = 0


        if len(gate_positions) > 0:
            for ptr in gate_positions:
                x, y = ptr
                if not self.pass_first_line:
                    if not top_line_contain_player and list(obs[y, x, :]) == [214, 92, 92]:
                        top_line_contain_player = True
                else:
                    if not bot_line_contain_player and list(obs[y + self.gate_height, x, :]) == [214, 92, 92]:
                        bot_line_contain_player = True
                if visualize_line:
                    obs[y, x, :] *= 0
                    obs[y + self.gate_height, x, :] *= 0    
        else:
            for ptr in final_gate_positions:
                x, y = ptr
                if not self.pass_first_line:
                    if not top_line_contain_player and list(obs[y, x, :]) == [214, 92, 92]:
                        top_line_contain_player = True
                else:
                    if not bot_line_contain_player and list(obs[y + self.gate_height, x, :]) == [214, 92, 92]:
                        bot_line_contain_player = True
                if visualize_line:
                    obs[y, x, :] *= 0
                    obs[y + self.gate_height, x, :] *= 0
        
        
        if self.prev_gate_y == gate_y:
            self.same_gate_y_cnt += 1
            if self.same_gate_y_cnt > 20:
                # fall here. add penalty
                # reward += self.fall_down_reward
                if bot_line_contain_player or top_line_contain_player:
                    self.fall_between_gate_set.add(self.gate_ind)
                    reward += self.fall_down_between_gate_reward
                    #print('fall down between gate', self.gate_ind)
                    if self.keep_info:
                        info['fall down between gate'] = 1
                else:
                    reward += self.fall_down_reward
                    #print('fall down near gate', self.gate_ind)
                    if self.keep_info:
                        info['fall down near gate'] = 1
        else:
            self.same_gate_y_cnt = 0

        if not self.pass_first_line:
            if self.prev_top_line_contain_player:
                if not top_line_contain_player and self.gate_ind not in self.fall_between_gate_set:
                    self.pass_first_line= True
        else:
            if self.prev_bot_line_contain_player:
                if not bot_line_contain_player and self.gate_ind not in self.fall_between_gate_set:
                    if self.gate_ind not in self.pass_gate_set:
                        self.pass_gate_set.add(self.gate_ind)
                        # print('pass gate', self.gate_ind)
                        # pass here, add reward
                        reward += self.pass_gate_reward
                        if self.keep_info:
                            info['pass gate'] = 1
                    
        self.prev_top_line_contain_player = top_line_contain_player
        self.prev_bot_line_contain_player = bot_line_contain_player
        
        if self.prev_gate_y < gate_y:
            self.gate_ind += 1
            self.pass_first_line = False
            if self.gate_ind > self.max_gate:
                terminated = True
        # print(gate_y - self.prev_gate_y)
        self.prev_gate_y = gate_y
        # print('t, reward, pass_gate_set, gate_y, gate_ind', t, reward, self.pass_gate_set, gate_y, self.gate_ind)

        return obs, reward, terminated, info

    def reset(self, **kwargs):
        """Resets the environment with `kwargs`."""

        self.pass_gate_set = set([])
        self.prev_top_line_contain_player = False
        self.prev_bot_line_contain_player = False
        self.prev_gate_y = 0
        self.same_gate_y_cnt = 0
        self.gate_ind = 0
        self.pass_first_line = False
        self.fall_between_gate_set = set([])
        
        
        return self.env.reset(**kwargs)


class TestWrapperB(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        # print('obs.shape', obs.shape)
        print('reward', reward)
        return obs, reward, terminated, info



class SaveStudentData(gym.Wrapper):
    # for atari game we need to save raw data.
    def __init__(self, env):
        super().__init__(env)
        self.max_data_amount = 5000
        self.max_full_data_cnt = 400
        self.data = {'obs': [], 'a': [], 'r': []}
        self.path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/'
        
        for key in self.data:
            full_path = self.path + key + '.npy'
            if path.exists(full_path):
                os.remove(full_path)
        self.finished_collecting_data = False

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.curr_obs = obs
        return obs

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)

        if not self.finished_collecting_data:
            if len(self.data['r']) < self.max_data_amount:
                self.data['obs'].append(self.curr_obs)
                self.data['a'].append(action)
                self.data['r'].append(reward)
            else:
                print('save student data!!!!')
                # Load the existing data
                self.save_or_append_data('obs', self.data['obs'])
                self.save_or_append_data('a', self.data['a'])
                self.save_or_append_data('r', self.data['r'])

                self.data =  {'obs': [], 'a': [], 'r': []}
                self.finished_collecting_data = True
        
        self.curr_obs = obs
        return obs, reward, terminated, info
    
    def save_or_append_data(self, filename, new_data):
        full_path = self.path + filename
        if not path.exists(full_path + '.npy'):
            # # File exists, load the existing data
            # existing_data = list(np.load(full_path + '.npy'))
            # if len(existing_data) < self.max_full_data_cnt: 
            # # Concatenate the new data with the existing data
            #     updated_data = existing_data + new_data
            #     # updated_data = np.concatenate((existing_data, new_data))
            #     # Save the updated array
            #     np.save(full_path, updated_data)
        # else:
        #     # File does not exist, save directly
            np.save(full_path, new_data)







from world_model.train_rewardNN import load_data_from_batch
from world_model.train_rewardNN import CNNRewardModel
from world_model.train_rewardNN import train_model
from world_model.train_rewardNN import WeightedMSELoss
from world_model.train_rewardNN import test_rewardNN
from world_model.train_error_correction import TransformerModel
from world_model.train_error_correction import exp_train_error_model
import torch

class TestWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/'
    
        loaded_model_path = path + 'rewardNN_ski.pth'
        
        self.model = CNNRewardModel()
        self.model.load_state_dict(torch.load(loaded_model_path))
        self.save_model_path = 'rewardNN_finetuned.pth'
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.normalization = 1
        self.model.eval()

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.curr_obs = obs
        return obs

    def step(self, action):
        tensor_obs = torch.tensor(self.curr_obs.transpose(2, 0, 1) / self.normalization, dtype=torch.float32).unsqueeze(0)
        tensor_action = torch.tensor([action/ self.normalization], dtype=torch.long)
        tensor_obs, tensor_action = tensor_obs.to(self.device), tensor_action.to(self.device)
        
        predicted_reward = self.model(tensor_obs, tensor_action).item()

        obs, reward, terminated, info = self.env.step(action)
        self.curr_obs = obs
        # print('obs.shape', obs.shape)
        if abs(reward) > 10: 
            print('reward', reward, 'predicted_reward', predicted_reward)
        return obs, reward, terminated, info






class UpdateRewardNN(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.variable = 0
        self.batches = []

        self.path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/'
    
        loaded_model_path = self.path + 'rewardNN_ski.pth'
        
        self.model = CNNRewardModel()
        self.model.load_state_dict(torch.load(loaded_model_path))
        self.save_model_path = 'rewardNN_finetuned.pth'
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.normalization = 1
        self.model.eval()


        self.error_model_path = 'error_correction.pth' 
        self.error_model = TransformerModel()
        # self.error_model.load_state_dict(torch.load(error_model_path))
        self.error_model.to(self.device)
        self.error_model_finished_train = False

        
        self.does_update = False
        print ('----------------------------------------------------------')
        print('finish init!!!!!!!!')

         

    def update_counter(self, addition):
        self.variable += addition

    def collect_data(self, batch):
        self.batches.append({'obs': batch['obs'], 
                             'actions': batch['actions'], 
                             'rewards': batch['rewards']})
        if len(self.batches) > 5:
            self.does_update = True
    
        
    
    def train_error_correction_nn(self):
        print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print ('train error correction nn')

        X = []
        errors = []
        cnt = 0
        for batch in self.batches:
            batch_size = batch['rewards'].numel()
            for i in range(batch_size): 
                o = batch['obs'][i].cpu().numpy()
                r = batch['rewards'][i].item()
                a = batch['actions'][i].item()
                print('o.shape', o.shape, 'a', a, 'r', r)
                X.append((o, a)) 
                
                obs_, action_ = o, a
                
                tensor_obs = torch.tensor(obs_.transpose(2, 0, 1) / self.normalization, dtype=torch.float32).unsqueeze(0)
                tensor_action = torch.tensor([action_ /self.normalization], dtype=torch.long)
                tensor_obs, tensor_action = tensor_obs.to(self.device), tensor_action.to(self.device)
                
                predicted_reward = self.model(tensor_obs, tensor_action).item()
                errors.append(predicted_reward - r)
                
                cnt += 1
        # X = []
        # errors = []
        # cnt = 0

        # obs_npy = np.load(self.path + 'obs.npy')
        # r_npy = np.load(self.path + 'r.npy')
        # a_npy = np.load(self.path + 'a.npy')
       
        # for i in range(len(a_npy)): 
        #     o = obs_npy[i, :]
        #     r = r_npy[i]
        #     a = a_npy[i]
        #     # print('o.shape', o.shape, 'a', a, 'r', r)
        #     X.append((o, a)) 
            
        #     obs_, action_ = o, a
            
        #     tensor_obs = torch.tensor(obs_.transpose(2, 0, 1) / self.normalization, dtype=torch.float32).unsqueeze(0)
        #     tensor_action = torch.tensor([action_ /self.normalization], dtype=torch.long)
        #     tensor_obs, tensor_action = tensor_obs.to(self.device), tensor_action.to(self.device)
            
        #     predicted_reward = self.model(tensor_obs, tensor_action).item()
        #     # print('predicted_reward', predicted_reward)
        #     # print('predicted_reward - r', predicted_reward - r)
        #     errors.append(predicted_reward - r)
            
        #     cnt += 1
        # np.save(self.path + 'student_X', X)
        # np.save(self.path + 'student_errors', errors)

        X = [(torch.tensor(obs.transpose(2, 0, 1)/self.normalization, dtype=torch.float32),
                torch.tensor(action/self.normalization, dtype=torch.long))
                for obs, action in X]
        
        Y = torch.tensor(errors, dtype=torch.float32)

        self.error_model = exp_train_error_model(X, Y, self.error_model, save_path=self.error_model_path)  # Ensure this is the same architecture as the trained one
        self.error_model.to(self.device)
        self.error_model.eval()
        print ('saved the error model!')
        print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            
            

    def report_stats(self):  
        pass
    
    def clean_batch(self):
        self.batches = []
        self.batch_error_counter = 0
        self.batch_all_counter = 0
        self.batch_tp = 0
        self.batch_fp = 0
        self.batch_tn = 0
        self.batch_fn = 0
    
        

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.curr_obs = obs
        return obs

    def step(self, action):
        if not self.error_model_finished_train and path.exists(self.error_model_path) :
            self.error_model.load_state_dict(torch.load(self.error_model_path))
            self.error_model.to(self.device)
            self.error_model.eval()
            self.error_model_finished_train = True
            print ('loaded error model weights for teacher workers')


        """Steps through the environment with `action`."""
        tensor_obs = torch.tensor(self.curr_obs.transpose(2, 0, 1) / self.normalization, dtype=torch.float32).unsqueeze(0)
        tensor_action = torch.tensor([action/ self.normalization], dtype=torch.long)
        tensor_obs, tensor_action = tensor_obs.to(self.device), tensor_action.to(self.device)
        
        predicted_reward = self.model(tensor_obs, tensor_action).item()
        predicted_error = self.error_model(tensor_obs, tensor_action).item()
        predicted_reward = predicted_reward - predicted_error

        obs, reward, terminated, info = self.env.step(action)
        self.curr_obs = obs

        
        reward = predicted_reward
                

        return obs, reward, terminated, info
