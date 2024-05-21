import model.util
import numpy as np
import pickle
import torch
import torch.nn.functional as F

from enum import Enum

from ray.rllib.utils.torch_utils import convert_to_torch_tensor

class ModelType(Enum):
    TORCH = 1
    TREE = 2


class ModelWrapper():
    def __init__(self, model_type, model=None):
        self.model_type = model_type
        self.model = model

    def load(self, import_path, action_space, observation_space, config):
        if self.model_type == ModelType.TORCH:
            self.model = model.util.load_torch_model(import_path, action_space, observation_space, config)
            print('!!!!!!!!!!!!!!!!!!!! loaded torch model')
        elif self.model_type == ModelType.TREE:
            self.model = pickle.load(open(import_path, 'rb'))

    def set(self, in_model):
        self.model = in_model

    def get_action(self, obs, get_decision_path = False):
        

        if self.model_type == ModelType.TORCH:
            
            # Notice that conflict with environments
            # also need a more elegant handling in future
            # here please make changes automatically
            name = 'usar'
            if name == 'ski' or name == 'pacman':
                obs = self._preprocess_obs(obs)
                # print(obs.shape)
                action_logit = self.model({"obs": torch.unsqueeze(obs, 0)})[0]
                action_prob = F.softmax(action_logit, dim=1).cpu().detach().numpy()

                log_action_prob = np.log(action_prob)
                action = np.random.choice(len(action_prob[0]), p=action_prob[0])

            elif name == 'usar':

                tensor_obs = convert_to_torch_tensor(obs, device=next(self.model.parameters()).device)

                for key in tensor_obs.keys():
                    tensor_obs[key] = torch.unsqueeze(tensor_obs[key], 0)        
                action_logit = self.model.forward({"obs": tensor_obs}, None, None)[0].squeeze(0)
                action_prob = F.softmax(action_logit, dim=0).cpu().detach().numpy()
                # action = np.random.choice(len(action_prob), p=action_prob)
                action = np.random.choice(len(action_prob), p=action_prob)
                log_action_prob = np.log(action_prob)
            else:
                obs = self._preprocess_obs(obs)
                action_logit = self.model({"obs": torch.unsqueeze(obs, 0)})[0]
                action_prob = F.softmax(action_logit, dim=1).cpu().detach().numpy()

                log_action_prob = np.log(action_prob)
                action = np.random.choice(len(action_prob[0]), p=action_prob[0])

            # Use maximum entropy formulation to estimate q values
            importance = np.max(log_action_prob) - np.min(log_action_prob)
            return action, action_prob, importance
        elif self.model_type == ModelType.TREE:
            obs = self._preprocess_obs(obs)
            # print ('tree obs', obs)
            # action = self.model.predict([obs])[0]
            action_prob = self.model.predict_proba([obs])[0]
            importance = None
            action = np.random.choice(len(action_prob), p=action_prob)

            
            
            if not get_decision_path:
                return action, action_prob, importance
        
            else:
                # Get the decision path of the observation
                decision_path = self.model.decision_path([obs]).toarray()[0]
                # print ('decision_path', decision_path)
                feature = self.model.tree_.feature
                threshold = self.model.tree_.threshold
                
                # Print the decision path
                node_indicator = self.model.decision_path([obs])
                leaf_id = self.model.apply([obs])
                
                # Obtain ids of the nodes `samples` goes through, i.e., row `id`.
                node_index = node_indicator.indices[node_indicator.indptr[0]:
                                                    node_indicator.indptr[1]]
                
                # print('Rules used to predict sample {}:'.format(obs))
                # print ('obs', obs)

                '''wall_features, 
                    ghost_features,
                    ghost_far_features, 
                    food_features, 
                    ghost_mode,
                    capsule_features
                '''
                feature_names = {0: {0: 'North has no wall.', 
                                    1: 'North has a wall.'},
                                1: {0: 'South has no wall.', 
                                    1: 'South has a wall.'},
                                2: {0: 'East has no wall.', 
                                    1: 'East has a wall.'},
                                3: {0: 'West has no wall.', 
                                    1: 'West has a wall.'},
                                4: {0: 'No nearby ghost in the North.', 
                                    1: 'A nearby ghost is in the North.'},
                                5: {0: 'No nearby ghost in the South.', 
                                    1: 'A nearby ghost is in the South.'},
                                6: {0: 'No nearby ghost in the East.', 
                                    1: 'A nearby ghost is in the East.'},
                                7: {0: 'No nearby ghost in the West.', 
                                    1: 'A nearby ghost is in the West.'},
                                8: {0: 'Moving North gets away from the closest ghost.', 
                                    1: 'Moving North gets closer to the closest ghost.'},
                                9: {0: 'Moving South gets away from the closest ghost.', 
                                    1: 'Moving South gets closer to the closest ghost.'},
                                10: {0: 'Moving East gets away from the closest ghost.', 
                                    1: 'Moving East gets closer to the closest ghost.'},
                                11: {0: 'Moving West gets away from the closest ghost.', 
                                    1: 'Moving West gets closer to the closest ghost.'},
                                12: {0: 'Moving North gets away from the closest food.', 
                                    1: 'Moving North gets closer to the closest food.'},
                                13: {0: 'Moving South gets away from the closest food.', 
                                    1: 'Moving South gets closer to the closest food.'},
                                14: {0: 'Moving East gets away from the closest food.', 
                                    1: 'Moving East gets closer to the closest food.'},
                                15: {0: 'Moving West gets away from the closest food.', 
                                    1: 'Moving West gets closer to the closest food.'},
                                16: {0: 'A Ghost will not be scared for more than 0 seconds.', 
                                    1: 'A Ghost will be scared for more than 0 seconds.'},
                                17: {0: 'A Ghost will not be scared for more than 10 seconds.', 
                                    1: 'A Ghost will be scared for more than 10 seconds.'},
                                18: {0: 'A Ghost will not be scared for more than 20 seconds.', 
                                    1: 'A Ghost will be scared for more than 20 seconds.'},
                                19: {0: 'A Ghost will not be scared for more than 30 seconds.', 
                                    1: 'A Ghost will be scared for more than 30 seconds.'},
                                20: {0: 'Moving North gets away from the closest capsule.', 
                                    1: 'Moving North gets closer to the closest capsule.'},
                                21: {0: 'Moving South gets away from the closest capsule.', 
                                    1: 'Moving South gets closer to the closest capsule.'},
                                22: {0: 'Moving East gets away from the closest capsule.', 
                                    1: 'Moving East gets closer to the closest capsule.'},
                                23: {0: 'Moving West gets away from the closest capsule.', 
                                    1: 'Moving West gets closer to the closest capsule.'},
                                }
                path_str = ''
                for node_id in node_index:
                    # # Check if value of the split feature for the sample is below threshold
                    # if obs[feature[node_id]] <= threshold[node_id]:
                    #     threshold_sign = "<="
                    # else:
                    #     threshold_sign = ">"

                    # print("decision id node {} : (feature value = {}; threshold at node = {}) {} {}"
                    #     .format(node_id,
                    #             obs[feature[node_id]],
                    #             threshold[node_id],
                    #             threshold_sign,
                    #             threshold[node_id]))
                    
                    f = feature[node_id] # which feature correspond to
                    if f != -2:
                        f_v = obs[f] # value of the feauture
                        # print ('f_v', f_v)
                        f_name = feature_names[f][f_v]
                        path_str += 'feature id: ' + str(f) + ', value: ' + str(f_v) + ', meaning: ' + f_name + '\n'

                # Print the predicted action and its probability
                # print("Predicted action:", action)
                # print("Predicted action probability:", action_prob)

            return action, action_prob, importance, path_str

    def get_explanation(self, obs, action):
        explanation = None

        obs = self._preprocess_obs(obs)

        if self.model_type == ModelType.TREE:
            feature = self.model.tree_.feature
            threshold = self.model.tree_.threshold

            node_indicator = self.model.decision_path([obs])
            leaf_id = self.model.apply([obs])[0]

            # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
            node_index = node_indicator.indices[
                node_indicator.indptr[0] : node_indicator.indptr[1]
            ]

            explanation = []
            for node_id in node_index:
                # continue to the next node if it is a leaf node
                if leaf_id == node_id:
                    continue

                # check if value of the split feature for sample 0 is below threshold
                if obs[feature[node_id]] <= threshold[node_id]:
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"

                explanation.append({'node': node_id,
                            'feature': feature[node_id],
                            'value': obs[feature[node_id]],
                            'inequality': threshold_sign,
                            'threshold': threshold[node_id],
                            'is_leaf': False})

            explanation.append({'node': leaf_id,
                        'value': action,
                        'is_leaf': True})

        return explanation

    def _preprocess_obs(self, obs):
        
        if self.model_type == ModelType.TORCH:
            if type(obs) != torch.Tensor:
                obs = torch.tensor(obs, requires_grad = False)

        elif self.model_type == ModelType.TREE:
            if type(obs) == torch.Tensor:
                obs = obs.cpu().detach().numpy()

            # Flatten observation if it is multiple dimensions (excluding batch)
            if len(obs.shape) > 2:
                obs = obs.flatten()
        return obs