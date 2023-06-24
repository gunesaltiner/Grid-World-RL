from typing import List
from Environment import Environment
from abc import ABC, abstractmethod
import random


class RLAgent(ABC):
    action_size: int        # Number of possible actions
    state_size: int         # Number of possible state
    discount_rate: float    # Discount rate
    env: Environment        # Grid-World environment
    rnd: random.Random      # Random object

    def __init__(self, env: Environment, discount_rate: float, seed: int, action_size: int = 4):
        """
        Initiate the Agent
        :param env: The Environment where the Agent plays.
        :param discount_rate: Discount rate of cumulative rewards. Must be between 0.0 and 1.0
        :param action_size: Number of possible actions
        """
        self.env = env
        self.state_size = env.grid_size * env.grid_size

        assert action_size > 0, "Action size must be positive"
        self.action_size = action_size

        assert 0.0 <= discount_rate <= 1.0, "Discount rate must be in range [0.0, 1.0]"
        self.discount_rate = discount_rate

        self.rnd = random.Random(seed)

    @abstractmethod
    def train(self, **kwargs):
        """
        Implement this method, Not Call!

        You should implement this method for training of the corresponding RL approach.

        :param kwargs: Some methods need some more arguments.
        :return: None
        """

        pass

    @abstractmethod
    def act(self, state: int, is_training: bool) -> int:
        """
        Implement this method, Not Call!
        :param state: Current State as Integer (not Position)
        :param is_training: Some Agents may act differently during training.
        :return: Decided Action as int
        """

        pass

    def validate(self) -> (List[int], int):
        """
        This method returns the optimal list of action and the maximum total reward. The actions are decided by the
        agent after training

        :return: List of decided action and the maximum total reward
        """

        actions: List[int] = []
        total_reward: int = 0

        current_state: int = self.env.reset()
        done: bool = False
        max_iter = 100
        i = 0

        while not done and i < max_iter:
            # Decide action based on current state
            action = self.act(current_state, is_training=False)

            # Take action
            next_state, reward, done = self.env.move(action)

            # Update results
            total_reward += reward
            actions.append(action)

            # Update state
            current_state = next_state

            i += 1

        return actions, total_reward

