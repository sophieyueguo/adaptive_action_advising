from typing import Dict

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch


class LoggingCallbacks(DefaultCallbacks):

    #TODO: this is only for the student that needs advice. 

    def __init__(self):
        
        super().__init__()
        self.global_action_advice = 0
        self.global_follow_teacher = 0
        self.global_step = 0
        # self.worker_action_advice = {}
       
    
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        advised_policy = policies[DEFAULT_POLICY_ID]

        episode.custom_metrics["action_advice"] = advised_policy.action_advice
        episode.custom_metrics["follow_teacher"] = advised_policy.follow_teacher
        episode.custom_metrics["global_step"] = advised_policy.global_step

        # worker_id = worker.worker_index
        # episode.custom_metrics[str(worker_id) + "reward_length"] = advised_policy.reward_length

    '''Called in Main Trainer Driver''' 
    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        """Called at the beginning of Policy.learn_on_batch().

        Note: This is called before 0-padding via
        `pad_batch_to_sequences_of_same_size`.

        Also note, SampleBatch.INFOS column will not be available on
        train_batch within this callback if framework is tf1, due to
        the fact that tf1 static graph would mistake it as part of the
        input dict if present.
        It is available though, for tf2 and torch frameworks.

        Args:
            policy: Reference to the current Policy object.
            train_batch: SampleBatch to be trained on. You can
                mutate this object to modify the samples generated.
            result: A results dict to add custom metrics to.
            kwargs: Forward compatibility placeholder.
        """
        policy.update_teacher(train_batch)
        # TODO: separate updates
        policy.perform_inner_loop(train_batch)