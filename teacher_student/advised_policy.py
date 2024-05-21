import gym
import model.introspection_model as introspection_model
import model.model_wrapper as model_wrapper
import numpy as np
import torch
from typing import List, Type, Union

from . import util
import teacher_student.util
import config.config_loader as config_loader

from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import restore_original_dimensions, ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    convert_to_torch_tensor,
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.typing import ModelWeights, TensorType
from ray.tune.logger import pretty_print

from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID


import os
import random



def get_advised_policy(base_class):

    class AdvisedPolicy(base_class):
        def __init__(self, observation_space, action_space, config):
            self.advice_mode = config.get("advice_mode", False)
            
            self.action_advice = 0
            self.global_step = 0

            self.pretrained_model = config.get("pre_trained_model", None)
            
            if self.advice_mode == "never_advise":
                self.follow_teacher = 0
            
            else:
                self.teacher_initialized = False

                self.teacher_inner_trainer = None
                self.teacher_model = None
                
                # We only want to do this if we're in the driver policy!
                self.run_inner_loop = config.get("run_inner_loop", False)
                # student: run_inner_loop is true, teacher: run_inner_loop is false

                self.inner_loop_config = config.get("inner_loop_config", None)
                self.inner_loop_frequency = config.get("inner_loop_frequency", 5000)
                self.inner_loop_last_ts = 0

                self.teacher_config = config.get("teacher_model_path", None)

                self.advice_decay_rate = config.get("advice_decay_rate", 0)
                self.follow_teacher = config.get("follow_teacher_prob", 0)
                if self.advice_mode == "always_advise" or self.advice_mode == "adaptive_always_advise":
                    self.follow_teacher = 1
                elif self.advice_mode == "adaptive_decay_advise" or self.advice_mode == "decay_advise":
                    self.initial_follow_teacher_prob = config.get("initial_follow_teacher_prob", 1)
                    self.follow_teacher = self.initial_follow_teacher_prob
                
                

                self.burn_in = config.get("burn_in", 0)
                self.advising_max_step = config.get("advising_max_step", 0)
                self.step_count = 0
                self.advice_decay_step = config.get("advice_decay_step", 0)

                self.teacher_training_stop_timestep = config.get("teacher_training_stop_timestep", 0)
                self.teacher_training_iterations = config.get("teacher_training_iterations", 0)

                self.collect_student_data = False
                if self.advice_mode in ["adaptive_always_advise", "adaptive_decay_advise"]:
                    self.collect_student_data = True
                    self.adaptive_teacher_initial_advising_rate = config.get("adaptive_teacher_initial_advising_rate", 0)
                    self.adaptive_teacher_advising_period_steps = config.get("adaptive_teacher_advising_period_steps", 0)
                # self.collect_student_data = config.get("collect_student_data", False)
                
                self.use_linear_advice_decay = config.get("use_linear_advice_decay", False)

            super().__init__(observation_space, action_space, config)

            if self.advice_mode != "never_advise":
                teacher_model_config = config.get("teacher_model_config", config["model"])
                self.teacher_model = model_wrapper.ModelWrapper(model_wrapper.ModelType.TORCH)

                if self.teacher_config is not None:
                    self.teacher_model.load(self.teacher_config, action_space, observation_space, teacher_model_config)
                    self.teacher_model.model.to(self.device)
                    print ('***************************************loaded a teacher model')

            if self.pretrained_model != None:
                self.model.load_state_dict(torch.load(self.pretrained_model, map_location=next(self.model.parameters()).device))


        def _get_student_actions(self,
                                 input_dict,
                                 explore=None,
                                 timestep=None,
                                 episodes=None,
                                 **kwargs):
            return super().compute_actions_from_input_dict(
                input_dict,
                explore=explore,
                timestep=timestep,
                episodes=episodes,
                **kwargs)


        def _get_teacher_actions(self,
                                 input_dict,
                                 explore=None,
                                 timestep=None,
                                 episodes=None,
                                 **kwargs):
            student_model = self.model
            if self.teacher_model != None:
                self.model = self.teacher_model.model
            self.model.obs_space = student_model.obs_space
            weights_trainer_model = {k: v.cpu().detach().numpy() for k, v in self.model.state_dict().items()}
            weights = convert_to_torch_tensor(weights_trainer_model)


            teacher_actions, rnn_states, info_dict = super().compute_actions_from_input_dict(
                input_dict,
                explore=explore,
                timestep=timestep,
                episodes=episodes,
                **kwargs)

            self.model = student_model

            return teacher_actions, rnn_states, info_dict
  
        def compute_actions_from_input_dict(self,
                                            input_dict,
                                            explore=None,
                                            timestep=None,
                                            episodes=None,
                                            **kwargs):
            # Note that this method is typically called by the actors doing rollouts, not the driver doing batch updates.
            original_timestep = timestep

            # Be very cautious here! It's possible elements of the input dict are modified in-place by the below call, causing subsequent calls for teacher actions to be altered.
            # So if this occurs, we may need to deep copy the dict beforehand.
            student_actions, rnn_states, info_dict = self._get_student_actions(input_dict, explore, timestep, episodes, **kwargs)
            actions = student_actions

            if self.advice_mode != "never_advise":
                if self.run_inner_loop:
                    if self.advice_mode == "always_advise": 
                        teacher_actions, _, _ = self._get_teacher_actions(input_dict, explore, original_timestep, episodes, **kwargs)
                        actions = teacher_actions                        
                        info_dict["action_advice"] = torch.ones(len(teacher_actions))
            
                    elif self.advice_mode == "adaptive_always_advise":
                        if timestep != None: 
                            if timestep > self.burn_in: # temp usage - if after the time point, keep advising the students.
                                teacher_actions, _, _ = self._get_teacher_actions(input_dict, explore, original_timestep, episodes, **kwargs)
                                actions = teacher_actions                        
                                info_dict["action_advice"] = torch.ones(len(teacher_actions))
                            else:
                                info_dict["action_advice"] = torch.zeros(len(student_actions))

                    elif self.advice_mode == "decay_advise":
                        info_dict["action_advice"] = torch.zeros(len(actions))
                        if timestep != None:
                            if timestep < self.advising_max_step:
                                if timestep > self.burn_in: 
                                    if not self.use_linear_advice_decay:
                                        self.follow_teacher = self.initial_follow_teacher_prob * self.advice_decay_rate ** (timestep - self.burn_in)
                                    else:
                                        self.follow_teacher = max([0, self.initial_follow_teacher_prob - self.advice_decay_step * (timestep - self.burn_in)])
                                else:
                                    self.follow_teacher = 0
                                #self.follow_teacher = 1 - timestep * self.advice_decay_step
                                #self.follow_teacher = 1 - (timestep/1500000)*0.75

                            else:
                                self.follow_teacher = 0

                        if random.uniform(0, 1) < self.follow_teacher:
                            teacher_actions, _, _ = self._get_teacher_actions(input_dict, explore, original_timestep, episodes, **kwargs)
                            actions = teacher_actions                        
                            info_dict["action_advice"] = torch.ones(len(teacher_actions))
                        
                    
                    elif self.advice_mode == "adaptive_decay_advise":
                        info_dict["action_advice"] = torch.zeros(len(student_actions))
                        # if self.global_step < self.advising_max_step: 
                        if timestep != None: 
                            if timestep < self.advising_max_step: 
                            
                                if timestep < self.burn_in: # temp usage - if after the time point, keep advising the students.
                                    self.follow_teacher = self.adaptive_teacher_initial_advising_rate
                                else:     
                                    if not self.use_linear_advice_decay:
                                        self.follow_teacher = self.initial_follow_teacher_prob * self.advice_decay_rate ** (timestep - self.burn_in)
                                    else:
                                        self.follow_teacher = max([0, self.initial_follow_teacher_prob - self.advice_decay_step * (timestep - self.burn_in)])

                            else:
                                self.follow_teacher = 0
                    
                        if random.uniform(0, 1) < self.follow_teacher:
                            teacher_actions, _, _ = self._get_teacher_actions(input_dict, explore, original_timestep, episodes, **kwargs)
                            actions = teacher_actions                        
                            info_dict["action_advice"] = torch.ones(len(teacher_actions))

            return convert_to_numpy((actions, rnn_states, info_dict))


      
        def _update_target_teacher(self, train_batch):
            pass


        def update_teacher(self, postprocessed_batch):
            if self.advice_mode != "never_advise" and self.teacher_initialized:
                self.steps_processed += postprocessed_batch[SampleBatch.CUR_OBS].shape[0]

        # think loading weights can be buggy
        # '''Called in Main Trainer Driver''' 
        def get_weights(self) -> ModelWeights:
            
            if self.advice_mode in ["adaptive_always_advise", "adaptive_decay_advise"] and self.teacher_inner_trainer is not None:
                teacher_policy = self.teacher_inner_trainer.get_policy(DEFAULT_POLICY_ID)
                teacher_policy_model = teacher_policy.model
                return {
                    "model": {k: v.cpu().detach().numpy() for k, v in self.model.state_dict().items()},
                    "teacher_inner_trainer_model": {k: v.cpu().detach().numpy() for k, v in teacher_policy_model.state_dict().items()},
                }
            else:
                return {k: v.cpu().detach().numpy() for k, v in self.model.state_dict().items()}

        '''Called in RolloutWorker'''
        def set_weights(self, weights: ModelWeights) -> None:

            if self.advice_mode in ["adaptive_always_advise", "adaptive_decay_advise"] and 'teacher_inner_trainer_model' in weights:                
                model_weights = convert_to_torch_tensor(weights["model"], device=self.device)
                self.model.load_state_dict(model_weights)

                inner_teacher_weights = convert_to_torch_tensor(weights["teacher_inner_trainer_model"], device=self.device)
                self.teacher_model.model.load_state_dict(inner_teacher_weights)

            else:
                weights = convert_to_torch_tensor(weights, device=self.device)
                self.model.load_state_dict(weights)

        '''Called in Main Trainer Driver''' 
        # Will be called once for num_sgd_iter * (batch_size / minibatch_size).
        def perform_inner_loop(self, train_batch):
            if self.advice_mode != "never_advise":        
                if self.teacher_inner_trainer is None and self.run_inner_loop:
                    exp_config = config_loader.ConfigLoader.load_config(self.inner_loop_config, False)

                    self.teacher_inner_trainer = teacher_student.util.get_trainer(exp_config["alg"])(config=exp_config)
                    
                    if self.teacher_config is not None:
                        teacher_policy = self.teacher_inner_trainer.get_policy(DEFAULT_POLICY_ID)
                        teacher_policy_model = teacher_policy.model
                        teacher_policy_model.load_state_dict(torch.load(self.teacher_config, map_location=next(teacher_policy_model.parameters()).device))

                if self.advice_mode in ["adaptive_always_advise", "adaptive_decay_advise"]:
                    if self.collect_student_data and self.run_inner_loop:

                        if self.teacher_inner_trainer == None or (not self.teacher_inner_trainer.workers.local_worker().env.error_model_finished_train):
                            self.teacher_inner_trainer.workers.local_worker().env.collect_data(train_batch)
                        
                    if self.run_inner_loop and self.global_timestep - self.inner_loop_last_ts > self.inner_loop_frequency:
                        self.inner_loop_last_ts = self.global_timestep
                        
                        if self.teacher_training_stop_timestep < self.global_timestep < self.teacher_training_stop_timestep + self.adaptive_teacher_advising_period_steps:  
                         
                            print("*********** TRIGGERING AT " + str(self.global_timestep))
                            print ('********************************************************')
                            print ('********************************************************')
                            print ('********************************************************')
                            print ('********************************************************')
                            print ('********************************************************')
                            print ('********************************************************')
                            # the rewardnn is not directly shared among teacher workers. better to reload
                            self.teacher_inner_trainer.workers.local_worker().env.train_error_correction_nn()
                            self.teacher_inner_trainer.workers.local_worker().env.update_reward_nn()

                            for teacher_i in range(self.teacher_training_iterations):
                                print ('teacher_i', teacher_i)
                                result = self.teacher_inner_trainer.train()
                                print ('timestep', result['timesteps_total'], 'reward mean', result['episode_reward_mean'], 'length mean', result['episode_len_mean'])

                                # self.teacher_inner_trainer.workers.local_worker().env.report_stats()
                            
                                if teacher_i % 50 == 0 or teacher_i == self.teacher_training_iterations - 1:
                                    teacher_policy = self.teacher_inner_trainer.get_policy(DEFAULT_POLICY_ID)
                                    policy_model = teacher_policy.model
                                    torch.save(policy_model.state_dict(), 'teacher_policy.pth')
                            
                            
                        
                        teacher_policy = self.teacher_inner_trainer.get_policy(DEFAULT_POLICY_ID)
                        teacher_policy_model = teacher_policy.model
                        weights_teacher_inner_trainer_model = {k: v.cpu().detach().numpy() for k, v in teacher_policy_model.state_dict().items()}
                        inner_teacher_weights = convert_to_torch_tensor(weights_teacher_inner_trainer_model, device=self.device)
                        self.teacher_model.model.load_state_dict(inner_teacher_weights)

        '''Called in RolloutWorker'''                  
        def postprocess_trajectory(
            self, sample_batch, other_agent_batches=None, episode=None
        ):

            batch = super().postprocess_trajectory(sample_batch, other_agent_batches, episode)

            if self.advice_mode != "never_advise":
                batch.accessed_keys.add("action_advice")
                batch.accessed_keys.add("follow_teacher")
                batch.accessed_keys.add("global_step")

                # # # this computes for each worker            
                if 'action_advice' in batch:
                    self.action_advice += sum(batch['action_advice']) #batch['action_advice'].shape[0]

                if 't' in batch:
                    self.global_step += len(batch['t'])

                

            return batch
        

       
        
    return AdvisedPolicy


