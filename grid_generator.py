import os.path
import pickle
import random as rnd
from typing import Union
from Environment import Environment, Position


def manhattan_distance(a: Position, b: Position) -> int:
    """
        This method calculates Manhattan Distance between two given position.
    :param a: Point a: Position
    :param b: Point b: Position
    :return: a - b
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_closest_goal(starting_point: Position, grid: list) -> Position:
        goals = []

        for i in range(len(grid)):
            for j in range(len(grid)):
                if grid[i][j] == 'G':
                    goals.append([i, j])

        closest_goal = goals[0]

        for goal in goals:
            if manhattan_distance(goal, starting_point) < manhattan_distance(closest_goal, starting_point):
                closest_goal = goal

        return closest_goal


def generate(file_name: str, number_of_goals: int, number_of_death: int, size: int, mountain_rate: float = 0.5, min_distance: int = 8) -> Environment:
    """
        This method randomly generates an Environment object based on the given parameters. It also creates the grid
        file.
    :param file_name: File name without extension
    :param number_of_goals: Number of goals
    :param number_of_death: Number of deaths
    :param size: Size of grid
    :param mountain_rate: Rate of mountain. It must be in range [0.0, 1.0]
    :param min_distance: Minimum distance between the closest goal and the randomly selected starting point
    :return: Created Environment object
    """
    assert size > 0, "Invalid size"
    assert 0 <= mountain_rate <= 1, "Mountain rate must be in range [0.0, 1.0]"

    grid = [["" for _ in range(size)] for _ in range(size)]

    # Set goals
    for i in range(number_of_goals):
        while True:
            row = rnd.randrange(size)
            col = rnd.randrange(size)

            if grid[row][col] == "":
                grid[row][col] = 'G'
                break

    # Set death
    for i in range(number_of_death):
        while True:
            row = rnd.randrange(size)
            col = rnd.randrange(size)

            if grid[row][col] == "":
                grid[row][col] = 'D'
                break

    # Set mountains and flats
    for row in range(size):
        for col in range(size):
            if grid[row][col] != "":
                continue

            grid[row][col] = 'M' if rnd.random() < mountain_rate else 'F'

    # Set start index
    starting_pos = [0., 0.]
    while True:
        row = rnd.randrange(size)
        col = rnd.randrange(size)

        if grid[row][col] in ['F', 'M']:
            if manhattan_distance(get_closest_goal([row, col], grid), [row, col]) >= min_distance:
                starting_pos[0] = row
                starting_pos[1] = col
                break

    # Save
    data = {"grid": grid, "start": starting_pos}

    with open(f"{file_name}.pkl", "wb") as f:
        pickle.dump(data, f)

    env = Environment(f"{file_name}.pkl")

    env.save(file_name)

    return env


if __name__ == "__main__":
    if not os.path.exists("grid_worlds/"):
        os.mkdir("grid_worlds/")

    file_name = input("File Name: ")

    file_name = f"grid_worlds/{file_name}"

    number_of_goals = int(input("Number of goals: "))

    number_of_death = int(input("Number of deaths: "))

    size = int(input("Size of grid: "))

    mountain_rate = float(input("Mountain rate: "))

    min_distance = int(input("Minimum distance: "))

    env = generate(file_name, number_of_goals, number_of_death, size, mountain_rate, min_distance)

    print(env.starting_position)

    print(env)

