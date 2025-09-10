# Env Source Files
This directory contains the source files for the [Inspection Env](source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/inspection_env.py) the The Configuration files for the Inspection Env which configs some curriuculum learning parameters and other and Sensors configuration can be found [here](source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/inspection_cfg.py).

The Current Mapping Source code can be found [here](source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/map2d.py).

# Training Script
The training script for our env can be found [here](scripts/reinforcement_learning/skrl/train_cont_gru.py) for the continuous action space version and [here](scripts/reinforcement_learning/skrl/train_discrete_gru.py) for the discrete action space version.

We currently use the continuous action space version for training.

To just run the environment without training, you can use the script [here](scripts/environments/inspection_agent_cont.py)

In the main Source env you need to turn debug to True to see all the results of the systems