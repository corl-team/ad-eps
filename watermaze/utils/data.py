import os
import json
import numpy as np
from torchvision.transforms import v2

from collections import defaultdict
from torch.utils.data import IterableDataset
import h5py
from typing import List, Dict, Any
import torch


def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def load_learning_histories(runs_path: str) -> List[Dict[str, Any]]:
    learning_histories = []

    # There can be multiple runs of differnet algorithms (e.g. different seeds, training goals)
    for subdir, _, files in os.walk(runs_path):
        # Extract metadata for different learning histories
        for filename in files:
            if not filename.endswith(".metadata"):
                continue

            with open(os.path.join(subdir, filename)) as f:
                metadata = json.load(f)

            # Extract full learning history from chunks
            learning_history = {
                "states": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "timesteps": [],
                "mask": [],
                "goal": tuple(metadata["goal"]) if "goal" in metadata else None,
            }
            for filename in metadata["ordered_trajectories"]:
                with np.load(os.path.join(subdir, filename), allow_pickle=True) as chunk:
                    learning_history["states"].append(np.hstack(chunk["states"]))
                    learning_history["actions"].append(np.hstack(chunk["actions"]))
                    learning_history["rewards"].append(np.hstack(chunk["rewards"]))
                    # learning_history["mask"].append(chunk["mask"])
                    # learning_history["dones"].append(chunk["dones"])

            learning_history["states"] = np.hstack(learning_history["states"])
            learning_history["actions"] = np.hstack(learning_history["actions"])
            learning_history["rewards"] = np.hstack(learning_history["rewards"]).astype(
                np.float32
            )
            # learning_history["dones"] = np.vstack(learning_history["dones"])
            # learning_history["mask"] = np.vstack(learning_history["mask"]).astype(
            #     np.float32
            # )
            learning_histories.append(learning_history)

    return learning_histories


def subsample_history(learning_history, subsample):
    trajectories = []

    traj_data = defaultdict(list)
    for step in range(len(learning_history["dones"])):
        # append data
        traj_data["states"].append(learning_history["states"][step])
        traj_data["actions"].append(learning_history["actions"][step])
        traj_data["rewards"].append(learning_history["rewards"][step])

        if learning_history["dones"][step]:
            trajectories.append({k: np.array(v) for k, v in traj_data.items()})
            traj_data = defaultdict(list)

    subsampled_trajectories = trajectories[::subsample]
    subsampled_history = {
        "states": np.concatenate([traj["states"] for traj in subsampled_trajectories]),
        "actions": np.concatenate([traj["actions"] for traj in subsampled_trajectories]),
        "rewards": np.concatenate([traj["rewards"] for traj in subsampled_trajectories]),
    }
    return subsampled_history


class SequenceDataset(IterableDataset):
    def __init__(
        self,
        h5_path,
        seq_len,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.file = None

        with h5py.File(self.h5_path, "r") as f:
            self.length = len(f.keys())

    def __iter__(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        return iter(self._generate_data())

    def _generate_data(self):
        while True:
            idx = np.random.randint(self.length)
            goal_buffer = self.file[str(idx)]

            idx_seq = np.random.randint(0, len(goal_buffer["action"]) - self.seq_len)
            obs = goal_buffer["obs"][idx_seq : idx_seq + self.seq_len] / 255
            action = goal_buffer["action"][idx_seq : idx_seq + self.seq_len]
            reward = goal_buffer["reward"][idx_seq : idx_seq + self.seq_len]

            yield obs, action, reward

    def __del__(self):
        if self.file is not None:
            self.file.close()


def generate_goals(min_radius, n_goals, generator: np.random.Generator):
    alpha = generator.uniform(0, 2 * np.pi, size=n_goals)
    r = generator.uniform(min_radius, 500, size=n_goals)
    x = r * np.cos(alpha)
    y = r * np.sin(alpha)

    goals = np.stack((x, y)).T

    return goals
