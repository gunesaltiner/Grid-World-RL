"""
    Name: Güneş
    Surname: Altıner
    Student ID: S020893
"""

import os.path

import numpy as np

import rl_agents
from Environment import Environment
import time


GRID_DIR = "grid_worlds/"


if __name__ == "__main__":
    file_name = input("Enter file name: ")

    assert os.path.exists(os.path.join(GRID_DIR, file_name)), "Invalid File"

    env = Environment(os.path.join(GRID_DIR, file_name))

    seed = 300
    future_reward_discount_rate = 0.95
    epsilon = 1.0
    epsilon_rate_decay = 0.99
    minimum_epsilon_rate = 0.01
    alpha = 0.5
    maximum_episode = 100

    # Type your parameters
    agents = [
        rl_agents.QLearningAgent(env, seed, future_reward_discount_rate, epsilon, epsilon_rate_decay, minimum_epsilon_rate, alpha, maximum_episode),
        rl_agents.SARSAAgent(env, seed, future_reward_discount_rate, epsilon, epsilon_rate_decay, minimum_epsilon_rate, alpha, maximum_episode)]

    actions = ["UP", "LEFT", "DOWN", "RIGHT"]

    for agent in agents:
        print("*" * 50)
        print()

        env.reset()

        start_time = time.time_ns()

        agent.train()
        agent.plot_rewards()
        agent.plot_q_values()

        end_time = time.time_ns()

        path, score = agent.validate()

        print("Actions:", [actions[i] for i in path])
        print("Score:", score)
        print("Elapsed Time (ms):", (end_time - start_time) * 1e-6)

        print("*" * 50)
