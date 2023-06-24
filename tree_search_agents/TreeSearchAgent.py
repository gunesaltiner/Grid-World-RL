from abc import ABC, abstractmethod
from typing import List

from Environment import Environment


class TreeSearchAgent(ABC):
    @abstractmethod
    def run(self, env: Environment) -> (List[int], float, list):
        """
            This method takes the environment, and find the best (i.e., optimal) path from the starting point to any
            goal. This method returns not only the list of action but also the total score.
        :param env: Environment object
        :returns: List of action, total score and list of expansion
        """

        pass

    def play(self, env: Environment, actions: List[int]) -> float:
        """
            This method applies the given actions on the environment. Then, it returns the total score.
        :param env: Environment object
        :param actions: List of action
        :return: Total score
        """

        env.reset()
        total_score = 0

        for action in actions:
            _, score, done = env.move(action)

            total_score += score

            if done:
                break

        return total_score

    @property
    @abstractmethod
    def name(self) -> str:
        pass
