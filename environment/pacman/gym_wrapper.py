import gym

from . import featureExtractors
from . import ghostAgents
from . import layout
from . import pacman
from . import pacmanAgents

import model.model_wrapper as model_wrapper
import model.introspection_model as introspection_model
import teacher_student.fixedAdvise as fixedAdvise


class GymPacman(gym.Env):
    def __init__(self, env_config):
        self.rules = pacman.ClassicGameRules(multiagent = env_config["multiagent"])
        self.rules.quiet = True
        self.num_ghosts = 4
        self.multiagent = env_config["multiagent"]
        self.advice_budget = env_config["advice_budget"]
        self.advice_mode = env_config["advice_mode"]
        self.advice_strategy = env_config["advice_strategy"]
        self.introspection_decay_rate = env_config["introspection_decay_rate"]

        self.layout = layout.getLayout("originalClassic")
        

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(24,1,1),
            dtype=int
        )
       
        self.action_info = {
            "action_advice": 0,
            "action_introspection": 0,
            "action_student": 0
        }

        self.game = None
        self.feature_extractor = featureExtractors.VectorFeatureExtractor()

        
        self.teacher_model = None
        self.teacher_dt = None
        self.introspection_model = None
        self.fixed_advise_teacher = None
        self.introspection_prob = 1.0

        self.step_idx = 0
        
    def reset(self):
        if self.multiagent:
            pacman_agent = pacmanAgents.GreedyAgent()
            ghost_agents = [None for _ in range(self.num_ghosts)]
        else:
            ghost_agents = []
            pacman_agent = None
            for idx in range(self.num_ghosts):
                ghost_agents.append(ghostAgents.DirectionalGhost(idx + 1))

        # If there is already a running game, copy over the current introspection probability and budget so it gets carried forward
        if self.game is not None:
            self.advice_budget = self.game.advice_budget
            self.introspection_prob = self.game.introspection_prob

        self.game = self.rules.newGame(
            self.layout,
            pacman_agent,
            ghost_agents,
            self.teacher_model,
            self.teacher_dt,
            self.advice_budget,
            self.advice_mode,
            self.advice_strategy,
            self.introspection_model,
            self.introspection_prob,
            self.introspection_decay_rate,
            self.fixed_advise_teacher)

        features = self.feature_extractor.getUnconditionedFeatures(self.game.state)
        return features

    def step(self, action):
        return self.game.step(action)








from world_model.train_rewardNN import CNNRewardModel
from world_model.train_rewardNN import train_model
from world_model.train_error_correction import TransformerModel
from world_model.train_error_correction import exp_train_error_model
import torch
import os.path as path

class UpdateRewardNN(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.batches = []

        path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/'
    
        loaded_model_path = path + 'rewardNN_pacman.pth'
        
        self.model = CNNRewardModel()
        self.model.load_state_dict(torch.load(loaded_model_path, map_location=torch.device('cpu')))        
        self.save_model_path = path + 'rewardNN_finetuned.pth'
        #self.save_model_path = '/home/glow/workspace/aart-hri-repo/adaptive_action_advising/world_model/rewardNN_finetuned_10rubble.pth'
        
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        self.normalization = 1
        self.model.eval()

        print ('self.model initialized')


        self.error_model_path = 'error_correction.pth'
        self.error_model = TransformerModel()
        self.error_model.to(self.device)
        self.error_model_finished_train = False

        
        self.does_update = False


    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.curr_obs = obs
        return obs


    def collect_data(self, batch):
        self.batches.append({'obs': batch['obs'], 
                             'actions': batch['actions'], 
                             'rewards': batch['rewards']})
        if len(self.batches) > 10:
            self.does_update = True


    def train_error_correction_nn(self):
        print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print ('train error correction nn')

        X = []
        errors = []
        print ('len(self.batches)', len(self.batches))
        for batch in self.batches:
            batch_size = batch['rewards'].numel()
            # print ('batch_size', batch_size)
            for i in range(batch_size): 
                o = batch['obs'][i]
                r = batch['rewards'][i].item()
                a = batch['actions'][i].item()
                # print ('o, a, r', o, a, r)
                
                # print ('o', o)
                obs_, action_ = o, a 
                                
                obs_input = [f[0][0].item() for f in obs_]
                # print ('obs_input', obs_input)
                X.append((obs_input, a))
                tensor_obs = torch.tensor(obs_input, dtype=torch.float32).unsqueeze(0)
                tensor_action = torch.tensor([action_ /self.normalization], dtype=torch.long)
                tensor_obs, tensor_action = tensor_obs.to(self.device), tensor_action.to(self.device)
                
                predicted_reward = self.model(tensor_obs, tensor_action).item()
                errors.append(predicted_reward - r)
        print ('len(errors)', len(errors))
        print ('len(X)', len(X))
        print ('X[0]', X[0])

        # train_dataset, test_dataset = process_data(X, Y, data_type=data_type, normalization=1)
        X = [(torch.tensor(obs, dtype=torch.float32),
                torch.tensor(action/self.normalization, dtype=torch.long))
                for obs, action in X]
        
        Y = torch.tensor(errors, dtype=torch.float32)
        print ('len(Y)', len(Y))
        print ('Y[0]', Y[0])
        print ('len(X)', len(X))
        print ('X[0]', X[0])

        self.error_model = exp_train_error_model(X, Y, self.error_model, save_path=self.error_model_path)  # Ensure this is the same architecture as the trained one
        self.error_model.to(self.device)
        self.error_model.eval()
        print ('saved the error model!')
        print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')










    def step(self, action):

        if not self.error_model_finished_train and path.exists(self.error_model_path) :
            self.error_model.load_state_dict(torch.load(self.error_model_path))
            self.error_model.to(self.device)
            self.error_model.eval()
            self.error_model_finished_train = True
            print ('loaded error model weights for teacher workers')


            
        obs_input = [f[0][0] for f in self.curr_obs]
        # print ('obs_input', obs_input)

        tensor_obs = torch.tensor(obs_input, dtype=torch.float32).unsqueeze(0)
        tensor_action = torch.tensor([action/ self.normalization], dtype=torch.long)
        tensor_obs, tensor_action = tensor_obs.to(self.device), tensor_action.to(self.device)
        
        predicted_reward = self.model(tensor_obs, tensor_action).item()
        predicted_error = self.error_model(tensor_obs, tensor_action).item()
        predicted_reward = predicted_reward - predicted_error

        # print ('predicted_reward', predicted_reward)
        
        obs, reward, terminated, info = self.env.step(action)
        self.curr_obs = obs

        reward = predicted_reward

        return obs, reward, terminated, info




