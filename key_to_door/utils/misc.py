import os
import torch
import random
import numpy as np
import itertools


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def train_test_goals(grid_size, num_train_goals, seed):
    set_seed(seed)
    assert num_train_goals <= grid_size**2

    goals = np.mgrid[0:grid_size, 0:grid_size].reshape(2, -1).T
    goals = np.random.permutation(goals)

    train_goals = goals[:num_train_goals]
    test_goals = goals[num_train_goals:]
    return train_goals, test_goals


def train_test_goals_k2d(grid_size, num_train_goals, seed, num_test_goals=None):
    set_seed(seed)
    assert num_train_goals <= grid_size**4

    goals = np.mgrid[0:grid_size, 0:grid_size].reshape(2, -1).T
    goals = list(itertools.product(goals, goals))
    goals = np.random.permutation(goals)
    goals = goals[np.all(goals[:, 0, :] != goals[:, 1, :], axis=1)]

    train_goals = goals[:num_train_goals]
    test_goals = goals[num_train_goals:]

    if num_test_goals is not None:
        test_goals = test_goals[:num_test_goals]

    return train_goals, test_goals
