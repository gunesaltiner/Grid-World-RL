import os.path
import pickle as pkl
import copy
from typing import List, TypeVar, Union

import matplotlib.pyplot as plt

Position = TypeVar("Position", bound=List[int])


class Environment:
    def __init__(self, grid_file: str):
        """
        This method is the constructor of Environment class, which takes the grid file path to initiate the environment.
        Grid file is a pickle file, which contains two variables:
         - **grid**: 2 dimensional list where the nodes are annotated as a string.
         - **start**: Starting position of the agent.

        :param grid_file: The absolute/relative path of the file containing the corresponding informantion.
        :raise: File not found exception.
        """

        if not os.path.exists(grid_file):
            raise FileNotFoundError(f"The given grid_file is not found! ({grid_file}).")

        with open(grid_file, "rb") as f:
            _grid_data = pkl.load(f)

        self.grid = _grid_data["grid"]
        self.starting_position = _grid_data["start"]

        self.grid_size = len(self.grid)
        self.limits = [0, self.grid_size - 1]
        self.current_position = copy.deepcopy(self.starting_position)

    def reset(self) -> int:
        """
        This method resets the environment to the starting position.

        :return: Initial state index
        """
        self.current_position = copy.deepcopy(self.starting_position)

        return self.to_state(self.current_position)

    def to_state(self, position: Position) -> int:
        """
        This method converts a given position to state index.

        :param position: Target position
        :return: State index
        """
        return position[0] * self.grid_size + position[1]

    def to_position(self, state: int) -> Position:
        """
        This method converts a given state index to position.

        :param state: State index
        :return: Position
        """

        state_row = state // self.grid_size
        state_col = state % self.grid_size

        return [state_row, state_col]

    def set_current_state(self, state: int):
        """
        This method takes the index of a state as an input, then updates the current position accordingly.

        :param state: Index of the state
        :return: None
        :raise: Illegal state index exception
        """

        assert 0 <= state < self.grid_size * self.grid_size, "Illegal state index."

        state_row = state // self.grid_size
        state_col = state % self.grid_size

        self.current_position = [state_row, state_col]

    def move(self, action: int) -> (int, int, bool):
        """
        This method takes the decided action, then moves the agent accordingly. This method also provides the new
        **State**, corresponding **reward** value and whether the episode is **Done**, or not.

        The actions are defined as *Integer* values, as follows:
         - **UP**       :   0
         - **LEFT**     :   1
         - **DOWN**     :   2
         - **RIGHT**    :   3

        :param action: Taken action as an integer value in range ``[0, 3]``
        :returns: Tuple (**state**, **reward**, **done**) WHERE:
        **state** *(int)*: New state formed after the action is taken,
        **reward** *(int)*: Transition score,
        **done** *(bool)*: If the current episode is done, or not.
        :raise: Illegal action exception
        """

        assert 0 <= action <= 3, "Illegal action."

        if self.is_done(self.current_position):
            return self.to_state(self.current_position), 0, True

        new_position = self._move_vertical(action) if action in [0, 2] else self._move_horizontal(action)

        transition_reward = self.get_reward(self.current_position, new_position)
        done = self.is_done(new_position)

        self.current_position = [min(self.grid_size - 1, max(0, axis)) for axis in new_position]

        return self.to_state(self.current_position), transition_reward, done

    def _move_vertical(self, action: int) -> Position:
        """
        This method takes the action and moves the agent on vertical axis.

        :param action: Target action
        :return: Moved position
        """

        return [self.current_position[0] + (action - 1), self.current_position[1]]

    def _move_horizontal(self, action: int) -> Position:
        """
        This method takes the action and moves the agent on horizontal axis.

        :param action: Target action
        :return: Moved position
        """

        return [self.current_position[0], self.current_position[1] + (action - 2)]

    def get_node_type(self, position: Position) -> str:
        """
            This method provides the node on the grid based on the given position
        :param position: Target position
        :return: Corresponding node on the grid
        """

        return self.grid[position[0]][position[1]]

    def get_reward(self, previous_pos: Position, next_pos: Position) -> int:
        """
        This method applies the pre-defined reward function by taking previous and current positions of the agent. The
        score is determined depending on the **previous position** and **next position**, as seen in the table below:

        +------------+------------+---------+
        |  From      |  To        |  Reward |
        +============+============+=========+
        |  Flat      |  Flat      |  -1     |
        +------------+------------+---------+
        |  Flat      |  Mountain  |  -3     |
        +------------+------------+---------+
        |  Mountain  |  Mountain  |  -2     |
        +------------+------------+---------+
        |  Mountain  |  Flat      |  -1     |
        +------------+------------+---------+
        |   (Any)    |  Goal      |  100    |
        +            +------------+---------+
        |            |  Death     |  -100   |
        +            +------------+---------+
        |            |  Out       |  -1     |
        +------------+------------+---------+

        :param previous_pos: Previous coordinate as a list: ``[x, y]``
        :param next_pos: Next coordinate based on the taken action: ``[x, y]``

        :return: The reward
        :raise: Illegal position
        """

        assert previous_pos and len(previous_pos) == 2, "Illegal position"
        assert next_pos and len(next_pos) == 2, "Illegal position"

        if next_pos[0] < 0 or next_pos[0] >= self.grid_size:
            return -1
        if next_pos[1] < 0 or next_pos[1] >= self.grid_size:
            return -1

        if self.grid[next_pos[0]][next_pos[1]] == 'G':
            return 100
        if self.grid[next_pos[0]][next_pos[1]] == 'D':
            return -100
        if self.grid[next_pos[0]][next_pos[1]] == self.grid[previous_pos[0]][previous_pos[1]]:
            return -1 if self.grid[next_pos[0]][next_pos[1]] == 'F' else -2
        if self.grid[next_pos[0]][next_pos[1]] == 'M' and self.grid[previous_pos[0]][previous_pos[1]] == 'F':
            return -3

        return -1

    def is_done(self, position: Position) -> bool:
        """
        This method checks whether the given position ends the episode, or not.

        :param position: Given position
        :return: If the episode ends, or not
        :raise: Illegal position
        """

        assert position and len(position) == 2, "Illegal position"

        if position[0] < 0 or position[0] >= self.grid_size:
            return False
        if position[1] < 0 or position[1] >= self.grid_size:
            return False

        if self.grid[position[0]][position[1]] in ['G', 'D']:
            return True

        return False

    def save(self, file_name: str):
        """
            This method generates the corresponding Pickle file and visualize the grid world.
        :param file_name: Filename without extension
        :return: Nothing
        """
        with open(f"{file_name}.pkl", "wb") as f:
            data = {"grid": self.grid, "start": self.starting_position}

            pkl.dump(data, f)

        # Drawing
        colors = {
            "M": [85, 85, 85],
            "F": [170, 170, 170],
            "G": [0, 255, 0],
            "D": [255, 0, 0],
            "S": [154, 205, 50]
        }

        map = []

        for i in range(self.grid_size):
            map.append([])
            for j in range(self.grid_size):
                map[i].append(colors[self.grid[i][j]])

                if self.starting_position == [i, j]:
                    map[i][-1] = colors['S']

        plt.imshow(map, interpolation=None)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.starting_position[0] == i and self.starting_position[1] == j:
                    plt.text(j, i, f"{self.grid[i][j]} (S)", ha="center", color="black")
                else:
                    plt.text(j, i, self.grid[i][j], ha="center", color="black")

        plt.title("Grid Visualisation")

        plt.tight_layout()

        plt.savefig(f"{file_name}.png", dpi=1200, bbox_inches='tight')

        plt.close()

    def get_goals(self) -> List[int]:
        """
            This method finds the Goal nodes in the grid, then it returns as a list of state index.
        :return: List of state index of goals
        """

        goals = []

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] == 'G':
                    goals.append(self.to_state([i, j]))

        return goals

    def __str__(self):
        lines = ["\t".join(row) for row in self.grid]

        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.__str__().__hash__()
