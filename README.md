# GoHighFox

This repository contains an agent trained on the [ssbm_gym](https://github.com/Gurvan/ssbm_gym) environment using A2C.

## Requirements

First you need to install [ssbm_gym](https://github.com/Gurvan/ssbm_gym) following the instructions of the repository.

Then you need `pytorch` (tested with pytorch 1.2) for the training and inference, and `cloudpickle` for the vectorized environment if you want to train the agent.

Ubuntu 18.04 is supported for training. Linux and Windows are supported for visualizing/playing with the agent.

## Playing

You can play with the agent with `python play.py`.
In order to play you need a gamecube adapter with a controller plugged in port 2.
You can change the game duration by modifying `frame_limit` (default: 2 minutes).

## Training

Trianing is done with pytorch on a vectorized environment.
The algorithm is A2C with Generalized Advantage Estimation.

The agent is very limited and exists only for demonstration purpose.
It is not recurrent and observes only the current frame.
The bot still learns to go on the top platform and jump.

You can train with `python main.py`. See the code for optionnal arguments.
