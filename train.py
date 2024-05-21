import argparse
import config.config_loader
import environment.env_loader
import model.model_wrapper
import model.util
import numpy as np
import pickle
import ray
import teacher_student.util
import tempfile
import torch
import pickle

from datetime import datetime

from ray import tune
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.logger import pretty_print, UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR


from ray.rllib.utils.torch_utils import convert_to_torch_tensor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    default="train",
    choices=["train", "debug", "evaluate", "export", "experiments", "info", "world_model"],
    help="Running mode. Train to train a new model from scratch. Debug to run a minimal overhead training loop for debugging purposes."
         "Evaluate to rollout a trained pytorch model. Export to export a trained pytorch model from a checkpoint."
)
parser.add_argument(
    "--config",
    type=str,
    default="multi_grid_14room",
    help="Configuration file to use."
)
parser.add_argument(
    "--checkpoint-dir",
    type=str,
    default=None,
    help="Path to the directory containing an RLlib checkpoint. Used with the 'export' mode.",
)
parser.add_argument(
    "--import-path",
    type=str,
    default=None,
    help="Path to a pytorch saved model. Used with the 'evaluate' mode.",
)
parser.add_argument(
    "--export-path",
    type=str,
    default=None,
    help="Path to export a pytorch saved model. Used with the 'export' mode."
)
parser.add_argument(
    "--eval-episodes",
    type=int,
    default=1,
    help="Number of episodes to rollout for evaluation."
)
parser.add_argument(
    "--save-eval-rollouts",
    action="store_true",
    help="Whether to save (state, action, importance) triplets from evaluation rollouts.",
)
parser.add_argument(
    "--model-type",
    default="torch",
    choices=["torch", "tree"],
    help="The type of model to be imported. Options are 'torch' for a pytorch model or 'tree' for an sklearn tree classifier."
)
parser.add_argument(
    "--hpo",
    action="store_true",
    help="Whether to perform HPO during training."
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=100000,
    help="Maximum number of training iterations."
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=1,
    help="Number of samples to run per configuration."
)
parser.add_argument(
    "--max-concurrent",
    type=int,
    default=0,
    help="Maximum number of concurrent trials."
)


# A default logger to escape out ALE's stupid new ALE namespace which is forbidden by the windows file system
def default_logger(config):
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}_{}".format(str(config["alg"]), config["env"].replace("/", "_"), timestr)

    logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=DEFAULT_RESULTS_DIR)

    def default_logger_creator(config):
        """Creates a Unified logger with the default prefix."""
        return UnifiedLogger(config, logdir, loggers=None)

    return default_logger_creator


def rollout_episode(loaded_model, env, max_steps = 1000, flatten_obs = True, eval_transNN=False):
    obs = env.reset()
    done = False

    states = []

    step_idx = 0
    total_reward = 0

    bonus = False

    env_name = 'usar'
    if env_name == 'ski':
        data = [(None, None, env.unwrapped_obs)]
    elif env_name == 'pacman':
        # data = [env.game.state] # for visualization
        data = [(None, None, obs)]
    else:    
        data = [(None, None, obs)]

    # overried obs with transition network
    if eval_transNN:
        device = torch.device("cpu")
        from adaptive_action_advising.world_model.train_transNN import  CustomResNetWithEmbedding
        from adaptive_action_advising.world_model.train_transNN import visualize_comparison

        trans_model = CustomResNetWithEmbedding()
        model_path = 'world_model/transNN.pth'
        trans_model.load_state_dict(torch.load(model_path, map_location=device))
        trans_model = trans_model.to(device)
        trans_model.eval()
        normalization = 1
        step_counter = 0

    infos = []
    while not done:
        action, _, importance = loaded_model.get_action(obs)
        # print ('obs', obs)
        obs_flat = [f[0][0] for f in obs]

        states.append([obs_flat, action, importance])
        # states.append([obs, action, importance]) # for visualize

        if not eval_transNN:
            obs, reward, done, info = env.step(action)
            if env_name == 'pacman':
                # print ('state', env.game.state)
                # print ('reward', reward)
                # data.append(env.game.state) # for visualization
                pass
            if env_name == 'ski':                
                infos.append(info)
        else:
            # overried obs with transition network
            if step_counter > 8:
                break
            step_counter += 1

            
            gt_obs, reward, done, info = env.step(action)
            
            obs_torch = torch.tensor(obs['image'].transpose(2, 0, 1) / normalization, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
            action_torch = torch.tensor([action], dtype=torch.long).to(device)  # Add batch dimension

            with torch.no_grad():
                predicted_obs_tensor = trans_model(obs_torch, action_torch)
                predicted_obs_np = predicted_obs_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Remove batch dim and transpose
                predicted_obs_np = np.round(predicted_obs_np)

            # Update the observation dictionary with the new predicted observation
            visualize_comparison(gt_obs['image'], predicted_obs_np, figure_index=step_counter)
            gt_obs['image'] = predicted_obs_np
            obs = gt_obs
            
        if env_name == 'ski': 
            data.append((action, reward, env.unwrapped_obs))
        else:    
            data.append((action, reward, obs))
            # print ('action', action, 'reward', reward, 'obs', obs)

        if 'bonus' in info:
            bonus = info['bonus']
        else:
            if len(infos) > 0:
                bonus = {key: 0 for key in infos[0]}
                for info in infos:
                    for key in bonus:
                        bonus[key] += info[key]
                    


        step_idx += 1
        total_reward += reward

        if step_idx > max_steps:
            break
    
    final_reward = reward > 0

    if env_name == 'pacman':
        bonus = {}
        print ('Food Left', env.game.state.getNumFood())
        print ('num_capsule_eaten', env.game.state.data.num_capsule_eaten)
        print ('num_ghost_eaten', env.game.state.data.num_ghost_eaten)
        
        bonus['Food Left'] = env.game.state.getNumFood()
        bonus['num_capsule_eaten'] = env.game.state.data.num_capsule_eaten
        bonus['num_ghost_eaten']  = env.game.state.data.num_ghost_eaten
    env.reset()
    # if env_name == 'pacman':
    #     print ('Food Left', env.game.state.getNumFood())
    print ('total_reward', total_reward)
    print ()
    

    return states, step_idx, total_reward, bonus, final_reward, data


def rollout_episodes(loaded_model, env, num_episodes = 1, save_rollouts = False, max_steps = 600):
    all_episode_states = []
    num_steps = []
    rewards = []
    bonuses = []
    final_rewards = []

    for _ in range(num_episodes):
        states, steps, reward, bonus, final_reward, data = rollout_episode(loaded_model, env, max_steps = max_steps)
        # np.save('states', states) # uncomment if want to see the rollout states and make visualization.
        # print ('data', data)

        # with open('data.txt', 'w') as f:
        #     for d in data:
        #         f.write(str(d))
        #         f.write("\n\n")

        all_episode_states.append(states)
        num_steps.append(steps)
        rewards.append(reward)
        bonuses.append(bonus)
        final_rewards.append(final_reward)

    if save_rollouts:
        with open('model_trajectories.pickle', 'wb') as f:
            pickle.dump(all_episode_states, f)

    if True in bonuses or False in bonuses:
        mean_bonus, mean_final_rewarda = bonuses.count(True)/num_episodes, final_rewards.count(True)/num_episodes
    else:
        mean_bonus, mean_final_rewarda = None, None
        if len(bonuses) > 0:
            mean_bonus = {key: 0 for key in bonuses[0]}
            for bonus in bonuses:
                for key in mean_bonus:
                    mean_bonus[key] += bonus[key]
            mean_bonus = {key: mean_bonus[key]/num_episodes for key in mean_bonus}
            
    return np.mean(rewards), np.mean(num_steps), mean_bonus, mean_final_rewarda


def rollout_steps(loaded_model, env, num_steps = 600, max_steps = 1000, flatten_obs = True, reward_filter=0):
    steps_collected = 0

    all_episode_states = []

    while steps_collected < num_steps:
        states, steps, total_reward, _, _, _ = rollout_episode(loaded_model, env, max_steps = max_steps, flatten_obs = flatten_obs)
        if total_reward > reward_filter:
            all_episode_states.extend(states)
            steps_collected += steps

    all_episode_states = all_episode_states[:num_steps]
    print ('all_episode_states[0]', all_episode_states[0])

    return all_episode_states


def export_model(checkpoint_dir, export_path, exp_config):
    trainer = teacher_student.util.get_trainer(exp_config["alg"])(config=exp_config, logger_creator=default_logger(exp_config))

    trainer.restore(checkpoint_dir)

    policy = trainer.get_policy(DEFAULT_POLICY_ID)
    policy_model = policy.model

    torch.save(policy_model.state_dict(), export_path)


def train_model(mode = "train", checkpoint_dir = None, export_path = None, import_path = None):
    ray.init(local_mode = False)

    model.util.register_models()
    environment.env_loader.register_envs()

    exp_config = config.config_loader.ConfigLoader.load_config(args.config, args.hpo)

    model_type = model.model_wrapper.ModelType.TORCH
    if args.model_type == "tree":
        model_type = model.model_wrapper.ModelType.TREE

    if mode == "debug":
        trainer = teacher_student.util.get_trainer(exp_config["alg"])(config=exp_config, logger_creator=default_logger(exp_config))
        for i in range(100):
            result = trainer.train()

            print(result)
            
            if i == 0 or (i+1) % 10 == 0:
                exp_config = trainer.config
                env = environment.env_loader.env_maker(exp_config["env_config"], exp_config["env"])

                policy = trainer.get_policy(DEFAULT_POLICY_ID)
                policy_model = policy.model
                wrapped_model = model.model_wrapper.ModelWrapper(model_type, model=policy_model)
                                
                print ('agent training', i, 
                       'training reward mean', result['episode_reward_mean'],
                       'time step', result['timesteps_total'],
                       'advised issued', policy.action_advice)

    elif mode == "train":
        stop = {
            "timesteps_total": args.max_steps,
        }

        results = tune.run(
            teacher_student.util.get_trainer(exp_config["alg"]),
            config=exp_config,
            checkpoint_freq=200,
            checkpoint_score_attr="episode_reward_mean",
            keep_checkpoints_num=10,
            stop=stop,
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent if args.max_concurrent > 0 else None,
            checkpoint_at_end=True)

    elif mode == "experiments":
        stop = {
            "timesteps_total": args.max_steps,
        }

        results = tune.run(
            teacher_student.util.get_trainer(exp_config["alg"]),
            config=exp_config,
            checkpoint_freq=25,
            checkpoint_at_end=True,
            num_samples = 5,
            stop=stop)

    elif mode == "evaluate":
        if import_path is None:
            raise("import_path must be specified for the 'evaluate' mode.")

        exp_config = teacher_student.util.get_trainer(exp_config["alg"])(config=exp_config, logger_creator=default_logger(exp_config)).config
        env = environment.env_loader.env_maker(exp_config["env_config"], exp_config["env"])

        loaded_model = model.model_wrapper.ModelWrapper(model_type)
        loaded_model.load(import_path, env.action_space, env.observation_space, exp_config["model"])

        reward, steps, bonus, final_rewards = rollout_episodes(loaded_model, env, num_episodes=args.eval_episodes, save_rollouts=args.save_eval_rollouts)

        print("Evaluated {} episodes. Average reward: {}. Average num steps: {}".format(args.eval_episodes, reward, steps))
        print("Average bonus: {}. Average final reward: {}".format(bonus, final_rewards))

    elif mode == "export":
        if checkpoint_dir is None:
            raise("checkpoint_dir must be specified for the 'export' mode.")
        if export_path is None:
            raise("export_path must be specified for the 'export' mode.")

        export_model(checkpoint_dir, export_path, exp_config)

    elif mode == "info":
        if import_path is None:
            raise("import_path must be specified for the 'evaluate' mode.")
        
        env = environment.env_loader.env_maker(exp_config["env_config"], exp_config["env"])

        loaded_model = model.model_wrapper.ModelWrapper(model_type)
        loaded_model.load(import_path, env.action_space, env.observation_space, exp_config["model"])

        num_params = model.util.count_parameters(loaded_model.model)

        print(loaded_model.model)
        print("Num params: " + str(num_params))


    elif mode == "world_model":
        # collects data for the world_model using rollouts.
        # This happens when the teacher constructs its world.

        if import_path is None:
            raise("import_path must be specified for the 'evaluate' mode.")

        exp_config = teacher_student.util.get_trainer(exp_config["alg"])(config=exp_config, logger_creator=default_logger(exp_config)).config
        env = environment.env_loader.env_maker(exp_config["env_config"], exp_config["env"])

        loaded_model = model.model_wrapper.ModelWrapper(model_type)
        loaded_model.load(import_path, env.action_space, env.observation_space, exp_config["model"])

        # X, transNN_Y = [], []
        X, rewardNN_Y = [], []
        nonzero_reward_cnt = 0
        for idx in range(10000): 
            #400 for ski nonzeros
            #500 for usar
            print ('idx', idx)
            _, _, _, _, _, data = rollout_episode(loaded_model, env)
            obs = data[0][-1]
            # print ('obs', obs)
            for t in range(1, len(data)):
                action, reward, next_obs = data[t]

                # if reward > 100 or reward < -100:
                #     X.append((obs, action))
                #     # transNN_Y.append(next_obs)
                #     rewardNN_Y.append(reward)
                #     nonzero_reward_cnt += 1
                #     print ('nonzero reward counter', nonzero_reward_cnt)
                
                # only for pacman need to reshape the obs
                obs = [f[0][0] for f in obs]
                # obs = obs.reshape(24, 1)
                # print ('obs', obs, 'action', action, 'reward', reward)
            

                X.append((obs, action))
                rewardNN_Y.append(reward)

                obs = next_obs 
        
        # print ('X[0][0].shape, X[0][1], rewardNN_Y[0]', 
        #        X[0][0].shape,
        #        X[0][1], 
        #        rewardNN_Y[0])
        # print ('X[1][0].shape, X[1][1], rewardNN_Y[1]', 
        #        X[1][0].shape,
        #        X[1][1], 
        #        rewardNN_Y[1])
        # print (len(X), len(transNN_Y), len(rewardNN_Y))
        print ('len(X), len(rewardNN_Y)', len(X), len(rewardNN_Y))
        # print ('X[0]', X[0], 'rewardNN_Y[0]', rewardNN_Y[0])
        # np.save('world_model/X_nonzero_reward2', X)
        # np.save('world_model/X_random_reward', X)
        np.save('world_model/X_expert', X)
        # np.save('world_model/transNN_Y', transNN_Y)
        #np.save('world_model/rewardNN_Y_nonzero_reward2', rewardNN_Y)
        # np.save('world_model/rewardNN_Y_random_reward', rewardNN_Y)
        np.save('world_model/rewardNN_Y_expert', rewardNN_Y)

    ray.shutdown()


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    train_model(args.mode, args.checkpoint_dir, args.export_path, args.import_path)
