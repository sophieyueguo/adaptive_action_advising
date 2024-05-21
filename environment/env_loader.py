import environment.grid.gym_wrapper
import environment.grid.wrappers
import environment.grid_bidirection.gym_wrapper
import environment.grid_bidirection.wrappers
import environment.ski.wrappers
import environment.pacman.gym_wrapper

import gym
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from functools import partial
from ray.tune import register_env



# Use this function (with additional arguments if necessary) to additionally add wrappers to environments
def env_maker(config, env_name):
    env = None
    if env_name == "multi_grid":
        env = environment.grid.gym_wrapper.GymMultiGrid(config)
        env = environment.grid.wrappers.FullyObsWrapper(env)
        env = environment.grid.wrappers.ActionMasking(env)
        env = environment.grid.wrappers.PickupBonus(env)
    elif env_name == "multi_grid_teacher":
        env = environment.grid.gym_wrapper.GymMultiGrid(config)
        env = environment.grid.wrappers.FullyObsWrapper(env)
        env = environment.grid.wrappers.ActionMasking(env)
        env = environment.grid.wrappers.PickupBonus(env)
        env = environment.grid.wrappers.UpdateRewardNN(env)
    elif env_name == "multi_grid_bidirection":
        env = environment.grid_bidirection.gym_wrapper.GymMultiGrid(config)
        env = environment.grid_bidirection.wrappers.FullyObsWrapper(env)
        env = environment.grid_bidirection.wrappers.ActionMasking(env)
        env = environment.grid_bidirection.wrappers.PickupBonus(env)
    elif env_name == "multi_grid_bidirection_teacher":
        env = environment.grid_bidirection.gym_wrapper.GymMultiGrid(config)
        env = environment.grid_bidirection.wrappers.FullyObsWrapper(env)
        env = environment.grid_bidirection.wrappers.ActionMasking(env)
        env = environment.grid_bidirection.wrappers.PickupBonus(env)
        env = environment.grid_bidirection.wrappers.UpdateRewardNN(env)
    elif env_name == "Skiing-v4_stacked":
        env = gym.make("Skiing-v4")
        env = environment.ski.wrappers.RewardParaStudent(env)
        env = environment.ski.wrappers.ExtractFeature(env)
        env = environment.ski.wrappers.ObsWrapper(env)
        env = environment.ski.wrappers.SaveStudentData(env)
        env = wrap_deepmind(env, dim=84, framestack=True) #only for evaluating
    elif env_name == "Skiing-v4_stacked_teacher":
        env = gym.make("Skiing-v4")
        env = environment.ski.wrappers.RewardParaTeacher(env)
        env = environment.ski.wrappers.ExtractFeature(env)
        env = environment.ski.wrappers.ObsWrapper(env)
        # env = wrap_deepmind(env, dim=84, framestack=True) #only for evaluating
        #env = environment.ski.wrappers.TestWrapper(env)
        env = environment.ski.wrappers.UpdateRewardNN(env)
    elif env_name == "pacman":
        env = environment.pacman.gym_wrapper.GymPacman(config)
    elif env_name == "pacman_teacher":
        env = environment.pacman.gym_wrapper.GymPacman(config)
        env = environment.pacman.gym_wrapper.UpdateRewardNN(env)
    else:
        print ('env_name', env_name)
        raise("Unknown environment {}".format(env_name))
    return env


def register_envs():
    register_env("multi_grid", partial(env_maker, env_name = "multi_grid"))
    register_env("multi_grid_teacher", partial(env_maker, env_name = "multi_grid_teacher"))
    register_env("multi_grid_bidirection", partial(env_maker, env_name = "multi_grid_bidirection"))
    register_env("multi_grid_bidirection_teacher", partial(env_maker, env_name = "multi_grid_bidirection_teacher"))
    register_env("Skiing-v4_stacked", partial(env_maker, env_name = "Skiing-v4_stacked"))
    register_env("Skiing-v4_stacked_teacher", partial(env_maker, env_name = "Skiing-v4_stacked_teacher"))
    register_env("pacman", partial(env_maker, env_name = "pacman"))
    register_env("pacman_teacher", partial(env_maker, env_name = "pacman_teacher"))