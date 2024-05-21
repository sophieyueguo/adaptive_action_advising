(1) Build Conda Environment
Build environment using the yml file and change prefix if needed:
"conda env create -f <YOUR_REPO_PATH>/conda_env.yml"
Activate it.

(2) Setting up the Teacher
Change the teacher_model_path in config/<ENVIRONMENT>.yaml to your local absolute path:
"<YOUR_REPO_PATH>/saved_teacher_models/<TEACHER_TYPE>/model"

There are two teachers currently in this repo in the folder of "saved_teacher_models", and make sure the "teacher_model_path" is set to the correct path.
The default is a teacher that ignores rubble, but feel free to export a one by yourself.

To evaluate this teacher,
"python train.py --mode=evaluate --import-path=<MODEL_PATH>" --eval-episodes=1000

You shall see some statistics.
Note that for USAR, average bonus is for removing rubble and final reward is for finding the victim.

You can also evalute any model after the training by exporting a model
"python train.py --mode=export --checkpoint-dir=<CHECK_PTR_SAVED_IN_RAY_RESULTS>  --export-path=<MODEL_SAVED_PATH>"


(3) Run Student's Traning

First you may want to see how a student trains without advice.
Please set "advice_mode" to be "never_advise" in the config yaml.

Run

"python train.py  --mode=train --max-steps=7000000"

You should see the student performance growing.



Then you can checkout the teacher always advising by setting the mode on teacher's config. The performance is the teacher's.

Please set "advice_mode" to be "always_advise" in the config yaml.

Rerun the commands and you shall see the difference.
  


Similary, to run the advising, set "advice_mode" to be "decay_advise" in the config yaml.

The teacher gets adaptive if "teacher_training_stop_timestep" and "teacher_training_iterations" are set to be greater than 0. 

The default is the teacher getting adaptive with appropriate hyper-parameters.



(4) Collect data for the world model
python train.py --mode=world_model --import-path=<PATH>/adaptive_action_advising/saved_model/random --config=pacman

train NNs:
python world_model/train_rewardNN.py



(5) default is usar. make sure the env_name is correct all the places, and rewardNN is applicable.

