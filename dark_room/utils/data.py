import os
import json
import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset
from typing import List, Dict, Any


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
                "goal": tuple(metadata["goal"]) if "goal" in metadata else None
            }
            for filename in metadata["ordered_trajectories"]:
                chunk = np.load(os.path.join(subdir, filename))
                learning_history["states"].append(chunk["states"])
                learning_history["actions"].append(chunk["actions"])
                learning_history["rewards"].append(chunk["rewards"])
                learning_history["dones"].append(chunk["dones"])

            learning_history["states"] = np.vstack(learning_history["states"])
            learning_history["actions"] = np.vstack(learning_history["actions"])
            learning_history["rewards"] = np.vstack(learning_history["rewards"]).astype(np.float32)
            learning_history["dones"] = np.vstack(learning_history["dones"])

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
            trajectories.append(
                {k: np.array(v) for k, v in traj_data.items()}
            )
            traj_data = defaultdict(list)

    subsampled_trajectories = trajectories[::subsample]
    subsampled_history = {
        "states": np.concatenate([traj["states"] for traj in subsampled_trajectories]),
        "actions": np.concatenate([traj["actions"] for traj in subsampled_trajectories]),
        "rewards": np.concatenate([traj["rewards"] for traj in subsampled_trajectories])
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
        self._actions = np.concatenate([hist["actions"] for hist in histories]).flatten()
        self._rewards = np.concatenate([hist["rewards"] for hist in histories]).flatten()

        a = 0

    def __prepare_sample(self, start_idx):
        states = self._states[start_idx:start_idx + self.seq_len]
        actions = self._actions[start_idx:start_idx + self.seq_len]
        rewards = self._rewards[start_idx:start_idx + self.seq_len]

        return states, actions, rewards

    def __len__(self):
        return self._rewards.shape[0] - self.seq_len

    def __getitem__(self, idx):
        return self.__prepare_sample(idx)


class BCBuffer(Dataset):
    def __init__(self, runs_path: str, batch_size: int):
        histories = load_learning_histories(runs_path, bc=True)
        self._states = torch.as_tensor(
            np.concatenate([hist["states"] for hist in histories]), dtype=torch.long
        ).squeeze(-1)
        self._actions = torch.as_tensor(
            np.concatenate([hist["actions"] for hist in histories]), dtype=torch.long
        )
        self._rewards = torch.as_tensor(
            np.concatenate([hist["rewards"] for hist in histories])
        )

        self.batch_size = batch_size

    def sample_batch(self):
        idx = np.random.randint(0, len(self._states), size=self.batch_size)
        states = self._states[idx]
        actions = self._actions[idx]
        rewards = self._rewards[idx]

        return states, actions, rewards
