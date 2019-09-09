# Multiple Agents with Deep Reinforcement Learning

<img src="https://github.com/larryschirmer/deep_rl_multi_agent/raw/master/solved_multi_agent.gif" alt="solved multi agent" width="600"/>

## What is this Project

A deep reinforcement learning implementation of [ML Agents Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. The Tennis environment generates a playing field with two agents that can move in 2 degrees-of-freedom (up, down, left, right). The goal of both agents is to work together to volley the ball as many times as possible without dropping it. This project uses a feedforward network with a natural distribution output for each control. The agent learns to associate the mean of the distribution with the correct output for each state.

## How to Install

Because one of the project dependencies requires a specific version of tensorflow only available in python 3.5 and earlier, its easiest to use conda to build the environment for this project.

Run the following command to build this environment using the same dependancies I used:

```bash
conda env create -f environment.yml python=3.5
```

[See the conda docs for installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

As for the game engine, select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the project folder of this repository, and unzip (or decompress) the file.

## Getting Started

After installing this project's dependencies, launching `main.py` will begin training the model.

```bash
python main.py
```

This file launches the unity environment with twenty arms and begin posting to the console training updates every 100 epochs.

As an alternative, `Report.ipynb` run in a jupyter notebook is also a ducumented example of how to train this environment.

If you would like to run the solved model checkpoint that I have provided, launch a jupyter notebook environment:

```bash
jupyter notebook
```

and open `Tennis-Model-Eval.ipynb`


## Problem/Solution

![M-O](https://vignette.wikia.nocookie.net/pixar/images/3/32/M-o_wall%E2%80%A2e.png/revision/latest?cb=20110429103328)
**M-O - Pixar Wiki - Fandom.com**

Consider the problem where paths and equiptment need to be kept clean and there exists a robot specialized for the job. One robot can keep a workspace clean but easily gets distracted and overwhelmed when there is unexpected dirt tracks. How can two or more robots constructively work together to keep the space clean.

It is ambitious to make a perfect analogy with this problem and the Tennis environment, but it does present a small step towards making something like this a reality. Training multiple identical deep reinforcement learning agents in a shared state space with rewards for when they successfully work together is exactly what the Tennis environment sets out to do. The analogy is limited because the Tennis agents occupy seperate spaces in the shared state space so they can't share in the same work. It is possible to train the agents to work together to perform acts they could not do by themselves however.


## Important Files

- This `README.md`: describes the project and its files in detail
- `Tennis-Model-Eval.ipynb`: working demonstration of the model loaded after reaching training target
- `Report.ipynb`: document containing algorithms and methods, plots of training, and a discussion of future work
- `actor_critic.pt`: trained model checkpoint
- `main.py`: python file used to develop and train network
- `helpers.py`: collection of functions used to train, test, and monitor model
- `model.py`: class to return new model
- `scores.png`: a plot of scores and average scores from the 2 agents averaged together from each episode
- `losses.png`: model training loss over all of training


## The Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Help

Contributions, issues and feature requests are welcome.
Feel free to check [issues page](https://github.com/larryschirmer/deep_rl_multi_agent/issues) if you want to contribute.

## Author

- Larry Schirmer https://github.com/larryschirmer