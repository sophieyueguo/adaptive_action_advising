import gym
import gym_minigrid
import numpy as np

from gym import spaces
from gym.core import ObservationWrapper

# from environment.grid.bfs import BreadthFirstSearchPlanner


# A backwards compatibility wrapper so that RLlib can continue using the old deprecated Gym API
class GymCompatWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)

        return obs

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Since RLlib doesn't support the truncated variable (yet), incorporate it into terminated
        terminated = terminated or truncated

        return observation, reward, terminated, info


class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(self.env.width, self.env.height, 3),
                dtype="uint8"),
        })

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [gym_minigrid.minigrid.OBJECT_TO_IDX["agent"], gym_minigrid.minigrid.COLOR_TO_IDX["red"], env.agent_dir]
        )

        return {"image": full_grid}


class ActionMasking(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # The action mask sets a value for each action of either 0 (invalid) or 1 (valid).
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "action_mask": gym.spaces.Box(0.0, 1.0, shape=(self.action_space.n,))
        })

    def observation(self, obs):
        action_mask = np.ones(self.action_space.n)

        # Look at the position directly in front of the agent
        front_pos = self.unwrapped.front_pos
        front_pos_type = obs["image"][front_pos[0]][front_pos[1]][0]

        if front_pos_type == gym_minigrid.minigrid.OBJECT_TO_IDX["wall"]:
            action_mask[self.env.Actions.forward.value] = 0.0

        if front_pos_type != gym_minigrid.minigrid.OBJECT_TO_IDX["key"]:
            action_mask[self.env.Actions.pickup.value] = 0.0
        
        if front_pos_type != gym_minigrid.minigrid.OBJECT_TO_IDX["box"]:
            action_mask[self.env.Actions.pickup.value] = 0.0

        if front_pos_type != gym_minigrid.minigrid.OBJECT_TO_IDX["door"]:
            action_mask[self.env.Actions.toggle.value] = 0.0

        # Now disable actions that we intend to never use
        action_mask[self.env.Actions.drop.value] = 0.0
        action_mask[self.env.Actions.done.value] = 0.0

        if front_pos_type == gym_minigrid.minigrid.OBJECT_TO_IDX["ball"]:
            action_mask = np.zeros(self.action_space.n)
            action_mask[self.env.Actions.toggle.value] = 1.0

        return {**obs, "action_mask": action_mask}


class DoorUnlockBonus(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs = self.unwrapped.grid.encode()

        # If we just unlocked a door, add a reward shaping bonus.
        front_pos = self.unwrapped.front_pos
        front_pos_type = obs[front_pos[0]][front_pos[1]][0]
        front_pos_state = obs[front_pos[0]][front_pos[1]][2]

        
        if front_pos_type == gym_minigrid.minigrid.OBJECT_TO_IDX["door"] and front_pos_state == 2:
            is_locked_door = True
        else:
            is_locked_door = False

        obs, reward, done, info = self.env.step(action)
        
        
        bonus = 0.0
        if is_locked_door and action == self.env.Actions.toggle:
            front_pos_state = obs["image"][front_pos[0]][front_pos[1]][2]
            if front_pos_state == 0:
                bonus = 0.5

        reward += bonus

        return obs, reward, done, info


class ExplorationBonus(gym.Wrapper):
    """
    Adds an exploration bonus based the distance to the goal along a path.
    """

    def __init__(self, env):
        super().__init__(env)
        self.path = None
        self.path_idx = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        agent_pos = np.array(self.unwrapped.agent_pos)
        dist_to_current = np.linalg.norm(self.path[self.path_idx] - agent_pos)

        if self.path_idx < len(self.path) - 1:
            dist_to_next = np.linalg.norm(self.path[self.path_idx + 1] - agent_pos)

            if dist_to_next < dist_to_current:
                self.path_idx += 1
                
        if self.path_idx > 0:
            dist_to_prev = np.linalg.norm(self.path[self.path_idx - 1] - agent_pos)
            if dist_to_prev < dist_to_current:
                self.path_idx -= 1

            # print("Dist to prev: " + str(dist_to_prev))

        # The penalty is the remaining path length
        penalty = float(len(self.path) - self.path_idx)

        # Add penalty for distance from path
        penalty += float(np.linalg.norm(self.path[self.path_idx] - agent_pos))

        # Scale the penalty by the path length to fall between [0, 1]
        penalty /= len(self.path)
        # print(penalty)
        penalty /= self.max_steps

        reward -= penalty

        return obs, reward, done, info

    def _get_grid_obs(self):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            gym_minigrid.minigrid.OBJECT_TO_IDX['agent'],
            gym_minigrid.minigrid.COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        return full_grid

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        planner = BreadthFirstSearchPlanner()
        self.path = planner.plan(self._get_grid_obs())

        # Push the agent's starting position into the path so we can get an accurate counting of path length
        agent_pos = self.unwrapped.agent_pos
        self.path.insert(0, agent_pos)

        self.path_idx = 0

        return obs


class ActionBonus(gym.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    Example:
        >>> import miniworld
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ActionBonus
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> _, _ = env.reset(seed=0)
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> env_bonus = ActionBonus(env)
        >>> _, _ = env_bonus.reset(seed=0)
        >>> _, reward, _, _, _ = env_bonus.step(1)
        >>> print(reward)
        1.0
        >>> _, reward, _, _, _ = env_bonus.step(1)
        >>> print(reward)
        1.0
    """

    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited (state,action) pairs.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / np.sqrt(new_count)
        reward += bonus

        return obs, reward, terminated, info

    def reset(self, **kwargs):
        """Resets the environment with `kwargs`."""
        return self.env.reset(**kwargs)


class StateBonus(gym.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    Example:
        >>> import miniworld
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import StateBonus
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> _, _ = env.reset(seed=0)
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> env_bonus = StateBonus(env)
        >>> obs, _ = env_bonus.reset(seed=0)
        >>> obs, reward, terminated, truncated, info = env_bonus.step(1)
        >>> print(reward)
        1.0
        >>> obs, reward, terminated, truncated, info = env_bonus.step(1)
        >>> print(reward)
        0.7071067811865475
    """

    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited positions.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = tuple(env.agent_pos)

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / np.sqrt(new_count)
        reward += bonus

        return obs, reward, terminated, info

    def reset(self, **kwargs):
        """Resets the environment with `kwargs`."""
        return self.env.reset(**kwargs)





class PickupBonus(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        

    def step(self, action):
        obs = self.unwrapped.grid.encode()

        # Check if the agent is in front of a key and is about to pick it up
        front_pos = self.unwrapped.front_pos
        front_pos_type = obs[front_pos[0]][front_pos[1]][0]
        
        add_bonus = False
        ball_pos = (front_pos[0], front_pos[1])
        if front_pos_type == gym_minigrid.minigrid.OBJECT_TO_IDX["ball"]:
            if action == self.env.Actions.toggle and ball_pos not in self.ball_toggled:
                self.ball_toggled.add(ball_pos)
                obj_type = gym_minigrid.minigrid.Floor()
                self.env.grid.set(front_pos[0], front_pos[1], obj_type)
                add_bonus = True

        obs, reward, done, info = self.env.step(action)

        # bonus = 0.0
        if add_bonus:
            reward = max([0, self.rubble_reward - 0.9 * (self.step_count / self.max_steps)])  # Adjust the bonus value as needed
            #reward = 2
            info['bonus'] = True

        if done:
            # print('ball count', len(self.ball_toggled))
            pass
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        self.ball_toggled = set([])
        """Resets the environment with `kwargs`."""
        return self.env.reset(**kwargs)






    

from world_model.train_rewardNN import load_data_from_batch
from world_model.train_rewardNN import CNNRewardModel
from world_model.train_rewardNN import train_model
from world_model.train_rewardNN import WeightedMSELoss
from world_model.train_rewardNN import test_rewardNN
from world_model.train_error_correction import TransformerModel
from world_model.train_error_correction import exp_train_error_model
import torch
import os.path as path


class UpdateRewardNN(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.variable = 0
        self.batches = []

        # path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/'
    
        # loaded_model_path = path + 'rewardNN_usar_wgt10-15.pth'
        #loaded_model_path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/rewardNN_10rubble.pth'
        
        self.model = CNNRewardModel()
        # self.model.load_state_dict(torch.load(loaded_model_path))
        # self.save_model_path = path + 'rewardNN_finetuned.pth'
        self.save_model_path = 'rewardNN_finetuned.pth'
        #self.save_model_path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/rewardNN_finetuned_10rubble.pth'
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.normalization = 1
        self.model.eval()


        # self.error_model_path = 'error_correction.pth' #path + 'error_correction.pth'
        self.error_model = TransformerModel()
        # self.error_model.load_state_dict(torch.load(error_model_path))
        self.error_model.to(self.device)
        self.error_model_finished_train = False


        
        self.does_update = False

        # for evaluation
        self.error_counter = 0
        self.all_counter = 0
        self.batch_error_counter = 0
        self.batch_all_counter = 0
        self.batch_tp = 0
        self.batch_fp = 0
        self.batch_tn = 0
        self.batch_fn = 0

         

    def update_counter(self, addition):
        self.variable += addition

    def collect_data(self, batch):
        print ('calling collecting data')
        self.batches.append({'obs': batch['obs'], 
                             'actions': batch['actions'], 
                             'rewards': batch['rewards']})
        if len(self.batches) > 5:
            self.does_update = True

    def update_reward_nn(self):
        print ('!!!!!!!!!!!!!!!!!!!!!!!update reward nn')
        data_path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/'
        teacherX = np.load(data_path + 'X.npy', allow_pickle=True)
        teacherX = [(obs['image'], action)
            for obs, action in teacherX]
        teacherY = np.load(data_path + 'rewardNN_Y.npy', allow_pickle=True)


        if self.does_update:
            print ('self.batches')
            train_loader, test_loader, _ = load_data_from_batch(self.batches, X=list(teacherX), Y=list(teacherY))
            criterion = WeightedMSELoss(10, 15)
            self.model.train()
            self.model = train_model('rewardNN', train_loader, criterion, 
                    model=self.model, save_model=True, model_path=self.save_model_path, 
                    train_iter=100, lr=0.00001)
            error_counter = test_rewardNN(self.model, criterion, train_loader, None)
            print ('train error counter', error_counter)
            error_counter = test_rewardNN(self.model, criterion, test_loader, None)
            print ('test error counter', error_counter)
            
            self.model.eval()
        
    
    def train_error_correction_nn(self):
        print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print ('train error correction nn')

        X = []
        errors = []
        for batch in self.batches:
            batch_size = batch['rewards'].numel()
            for i in range(batch_size): 
                o = batch['obs'][i][7:].cpu().numpy().reshape(17, 17, 3)
                r = batch['rewards'][i].item()
                a = batch['actions'][i].item()
                X.append((o, a)) 
                
                obs_, action_ = o, a
                
                tensor_obs = torch.tensor(obs_.transpose(2, 0, 1) / self.normalization, dtype=torch.float32).unsqueeze(0)
                tensor_action = torch.tensor([action_ /self.normalization], dtype=torch.long)
                tensor_obs, tensor_action = tensor_obs.to(self.device), tensor_action.to(self.device)
                
                predicted_reward = self.model(tensor_obs, tensor_action).item()
                errors.append(predicted_reward - r)
                
        # train_dataset, test_dataset = process_data(X, Y, data_type=data_type, normalization=1)
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
        print ('all_counter', self.all_counter, 
                'error_counter', self.error_counter,
                # 'error rate', self.error_counter/self.all_counter,
                'batch_all_counter', self.batch_all_counter,
                'batch_error_counter', self.batch_error_counter,
                'batch_tp', self.batch_tp,
                'batch_fp', self.batch_fp,
                'batch_tn', self.batch_tn,
                'batch_fn', self.batch_fn)
    
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
        # if not self.error_model_finished_train and path.exists(self.error_model_path) :
        #     self.error_model.load_state_dict(torch.load(self.error_model_path))
        #     self.error_model.to(self.device)
        #     self.error_model.eval()
        #     self.error_model_finished_train = True
        #     print ('loaded error model weights for teacher workers')


        """Steps through the environment with `action`."""
        tensor_obs = torch.tensor(self.curr_obs['image'].transpose(2, 0, 1) / self.normalization, dtype=torch.float32).unsqueeze(0)
        tensor_action = torch.tensor([action/ self.normalization], dtype=torch.long)
        tensor_obs, tensor_action = tensor_obs.to(self.device), tensor_action.to(self.device)
        
        predicted_reward = self.model(tensor_obs, tensor_action).item()
        predicted_error = self.error_model(tensor_obs, tensor_action).item()
        # predicted_error = 0
        predicted_reward = predicted_reward - predicted_error

        obs, reward, terminated, info = self.env.step(action)
        self.curr_obs = obs

        
        # ###############count evaluation###############
        self.all_counter += 1
        self.batch_all_counter += 1 

        if reward == 0.501: # for rubble detection
            if predicted_reward <= 0.5:
                self.error_counter += 1
                self.batch_error_counter += 1
                self.batch_fn += 1
            else:
                self.batch_tp += 1
                        
        elif reward == 0:
            if predicted_reward > 0.5:
                self.error_counter += 1
                self.batch_error_counter += 1
                self.batch_fp += 1
            else:
                self.batch_tn += 1
            # predicted_reward = 0
        else:
            if predicted_reward < 0.5:
                self.error_counter += 1
                self.batch_error_counter += 1
                self.batch_fn += 1
            else:
                self.batch_tp += 1

        reward = round(predicted_reward)

        return obs, reward, terminated, info



