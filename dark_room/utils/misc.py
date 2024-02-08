import os
import torch
import random
import numpy as np


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def train_test_goals(grid_size, num_train_goals, seed):
    set_seed(seed)
    assert num_train_goals <= grid_size ** 2

    goals = np.mgrid[0:grid_size, 0:grid_size].reshape(2, -1).T
    goals = np.random.permutation(goals)

    train_goals = goals[:num_train_goals]
    test_goals = goals[num_train_goals:]
    return train_goals, test_goals