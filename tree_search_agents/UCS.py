"""
    Name:
    Surname:
    Student ID:
"""

from tree_search_agents.TreeSearchAgent import *
from tree_search_agents.PriorityQueue import PriorityQueue
import time


class UCSAgent(TreeSearchAgent):
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
                priority = score

                # Enqueue
                queue.enqueue({"state": next_state,
                               "total_reward": score,
                               "actions": actions + [action]},
                              priority)

            # Update visited states
            visited_states.add(state)

        # If no goal state is reached
        return [], 0, list(visited_states)

    @property
    def name(self) -> str:
        return "UCS"
