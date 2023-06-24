"""
    Name: Güneş
    Surname: Altıner
    Student ID: S020893
"""

from Environment import Environment
from rl_agents.RLAgent import RLAgent
import numpy as np
import random
import matplotlib.pyplot as plt


class SARSAAgent(RLAgent):
    epsilon: float  # Current epsilon value for epsilon-greedy
    epsilon_decay: float  # Decay ratio for epsilon
    epsilon_min: float  # Minimum epsilon value
    alpha: float  # Alpha value for soft-update
    max_episode: int  # Maximum iteration
    Q: np.ndarray  # Q-Table as Numpy Array

    rewards_per_episode: list

    def __init__(self, env: Environment, seed: int, discount_rate: float, epsilon: float, epsilon_decay: float,
                 epsilon_min: float, alpha: float, max_episode: int):
        """
        Initiate the Agent with hyperparameters.
        :param env: The Environment where the Agent plays.
        :param seed: Seed for random
        :param discount_rate: Discount rate of cumulative rewards. Must be between 0.0 and 1.0
        :param epsilon: Initial epsilon value for e-greedy
        :param epsilon_decay: epsilon = epsilon * epsilonDecay after all e-greedy. Less than 1.0
        :param epsilon_min: Minimum epsilon to avoid overestimation. Must be positive or zero
        :param max_episode: Maximum episode for training
        :param alpha: To update Q values softly. 0 < alpha <= 1.0
        """
        super().__init__(env, discount_rate, seed)

        assert epsilon >= 0.0, "epsilon must be >= 0"
        self.epsilon = epsilon

        assert 0.0 <= epsilon_decay <= 1.0, "epsilonDecay must be in range [0.0, 1.0]"
        self.epsilon_decay = epsilon_decay

        assert epsilon_min >= 0.0, "epsilonMin must be >= 0"
        self.epsilon_min = epsilon_min

        assert 0.0 < alpha <= 1.0, "alpha must be in range (0.0, 1.0]"
        self.alpha = alpha

        assert max_episode > 0, "Maximum episode must be > 0"
        self.max_episode = max_episode

        self.Q = np.zeros((self.state_size, self.action_size))

        # If you want to use more parameters, you can initiate below

        self.rewards_per_episode = []

    def train(self, **kwargs):
        """
        DO NOT CHANGE the name, parameters and return type of the method.

        You will fill the Q-Table with SARSA algorithm.

        :param kwargs: Empty
        :return: Nothing
        """
        for e in range(self.max_episode):
            current_state = self.env.reset()
            total_reward = 0
            while True:

                selected_action = self.act(current_state, True)
                new_state, reward, done = self.env.move(selected_action)

                total_reward += reward
                new_action = self.act(new_state, True)
                temp = reward + ((self.discount_rate * self.Q[new_state][new_action]) - self.Q[current_state][
                    selected_action])
                self.Q[current_state][selected_action] += self.alpha * temp
                new_position = self.env.to_position(new_state)
                self.env.set_current_state(new_state)
                current_state = new_state
                self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

                if done:
                    self.rewards_per_episode.append(total_reward)
                    break
        pass

    def act(self, state: int, is_training: bool) -> int:
        """
        DO NOT CHANGE the name, parameters and return type of the method.

        This method will decide which action will be taken by observing the given state.

        In training, you should apply epsilon-greedy approach. In validation, you should decide based on the Policy.

        :param state: Current State as Integer not Position
        :param is_training: If training use e-greedy, otherwise decide action based on the Policy.
        :return: Action as integer
        """

        selected_action = 0

        if is_training:
            generated_number = random.uniform(0, 1)
            if generated_number > self.epsilon:
                selected_action = np.argmax(self.Q[state])
            else:
                selected_action = self.rnd.randint(0, self.action_size - 1)
        else:
            selected_action = np.argmax(self.Q[state])

        return selected_action

    def plot_rewards(self):
        plt.plot(self.rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('SARSAAgent Training Performance')
        plt.show()

    def plot_q_values(self):
        max_q_values = np.max(self.Q, axis=1)  # Get maximum Q-value for each state
        optimal_actions = np.argmax(self.Q, axis=1)  # Get optimal action for each state

        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('States')
        ax1.set_ylabel('Max Q-values', color=color)
        ax1.plot(max_q_values, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'tab:red'
        ax2.set_ylabel('Optimal actions', color=color)
        ax2.plot(optimal_actions, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Max Q-values and Optimal actions for each state')
        fig.tight_layout()  # Otherwise the right y-label is slightly clipped
        plt.show()


