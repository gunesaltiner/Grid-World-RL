"""
    Name:
    Surname:
    Student ID:
"""

from tree_search_agents.TreeSearchAgent import *
from tree_search_agents.PriorityQueue import PriorityQueue
import time


class AStarAgent(TreeSearchAgent):
    def run(self, env: Environment) -> (List[int], float, list):
        # Initiate variables
        queue = PriorityQueue()

        visited_states = set()

        goals = env.get_goals()

        # Start from starting state
        queue.enqueue({"state": env.reset(), "total_reward": 0, "actions": []}, 0)

        while not queue.is_empty():
            # Next state that will be visited
            visit = queue.dequeue()

            state = visit["state"]
            total_reward = visit["total_reward"]
            actions = visit["actions"]

            # Visit once
            if state in visited_states:
                continue

            # If one of goal states is reached
            if state in goals:
                return actions, total_reward, list(visited_states)

            # Iterate over possible actions
            for action in range(4):
                env.set_current_state(state)

                next_state, reward, _ = env.move(action)

                # Do not enqueue if it has been already visited
                if next_state in visited_states:
                    continue

                # Calculate priority
                score = total_reward + reward
                heuristic = self.get_heuristic(env, next_state, goals=goals)
                priority = score + heuristic

                # Enqueue
                queue.enqueue({"state": next_state,
                               "total_reward": score,
                               "actions": actions + [action]},
                              priority)

            # Update visited states
            visited_states.add(state)

        # If no goal state is reached
        return [], 0, list(visited_states)

    def get_heuristic(self, env: Environment, state: int, **kwargs) -> float:
        """
            This method calculates the minimum Manhattan distance to goal states from the given state.

            Note that Heuristic value is negative minimum distance.
        :param env: Environment object
        :param state: Current State
        :param kwargs: Goal state
        :return: Negative Minimum Manhattan distance
        """

        assert "goals" in kwargs, "Goals must be provided for this heuristic function"

        goals = kwargs["goals"]

        current_position = env.to_position(state)

        distances = []

        # Iterate over each goal to find the minimum distance
        for goal in goals:
            goal_position = env.to_position(goal)

            # Manhattan Distance
            distance = abs(current_position[0] - goal_position[0]) + abs(current_position[1] - goal_position[1])

            distances.append(distance)

        # Heuristic = - minimum distance
        return -min(distances)

    @property
    def name(self) -> str:
        return "AStar"
