import os
import json
import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset
from typing import List, Dict, Any


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
        "actions": np.concatenate(
            [traj["actions"] for traj in subsampled_trajectories]
        ),
        "rewards": np.concatenate(
            [traj["rewards"] for traj in subsampled_trajectories]
        ),
    }
    return subsampled_history


class SequenceDataset(Dataset):
    def __init__(self, runs_path: str, seq_len: int = 60, subsample: int = 1):
        self.seq_len = seq_len

        print("Loading training histories...")
        histories = load_learning_histories(runs_path)
        self.goals = np.vstack([trajectory["goal"] for trajectory in histories])
        self.unique_goals = np.unique(self.goals, axis=0)

        if subsample > 1:
            histories = [subsample_history(hist, subsample) for hist in histories]

        self._states = np.concatenate([hist["states"] for hist in histories]).flatten()
        self._actions = np.concatenate(
            [hist["actions"] for hist in histories]
        ).flatten()
        self._rewards = np.concatenate(
            [hist["rewards"] for hist in histories]
        ).flatten()
        # self._mask = np.concatenate([hist["mask"] for hist in histories]).flatten()

        assert (
            self._states.shape[0]
            == self._actions.shape[0]
            == self._rewards.shape[0]
            # == self._mask.shape[0]
        )

    def __prepare_sample(self, start_idx):
        states = self._states[start_idx : start_idx + self.seq_len]
        actions = self._actions[start_idx : start_idx + self.seq_len]
        rewards = self._rewards[start_idx : start_idx + self.seq_len]
        # mask = self._mask[start_idx : start_idx + self.seq_len]

        return states, actions, rewards

    def __len__(self):
        return self._rewards.shape[0] - self.seq_len

    def __getitem__(self, idx):
        return self.__prepare_sample(idx)
