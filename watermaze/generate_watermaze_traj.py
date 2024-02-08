# warnings.simplefilter("ignore")

import gymnasium as gym

gym.logger.set_level(40)

from ppo_watermaze import WrapActionsMaze
import numpy as np
from sb3_contrib import RecurrentPPO
import deepmind_lab
import shimmy
from collections import defaultdict
import time
from tqdm import tqdm
import os
import pyrallis
from dataclasses import dataclass, asdict
import uuid
import copy
import wandb
import psutil
import gc
import h5py

import torch.multiprocessing as mp
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class GenConfig:
    project: str = "better-than"
    group: str = "traj_generation"
    name: str = "TRAJ-GEN"

    # model_weights: str = "../data/demonstrator/"
    model_weights: str = "demonstrator/"
    save_path: str = "trajectories/"
    hist_len: int = 10  # hist per goals it is
    hist_per_goal: int = 1
    num_goals: int = 2  # mod 22 == 0
    eps: float = 0.7
    n_proc: int = 22
    proportion_best: float = 0.1

    def __post_init__(self):
        self.name = (
            f"{self.name}-new-uncomp"
            f"-num_goals:{self.num_goals}"
            f"-hist_len:{self.hist_len}"
            f"-eps:{self.eps}"
            f"-hist_per_goal:{self.hist_per_goal}"
            f"-hist_per_goal:{self.proportion_best}"
            f"-{str(uuid.uuid4())[:8]}"
        )


class WrapObsMaze(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            shape=(3, 72, 96), low=0, high=255, dtype=np.uint8
        )

    def observation(self, obs):
        return obs["RGBD"][:3], obs["DEBUG.POS.TRANS"]


def run_collecting(model_path_goal, hist_len, max_eps, proportion_best):
    # print("HI")
    ts = time.time()
    model_path, goal, p_num = model_path_goal
    model, env = create_model_env(model_path, goal=goal)
    num_envs = 1
    buffer_glob = defaultdict(list)
    traj_counter = 0

    # print(f"INIT TOOK {time.time() - ts:.2f} sec")
    eps = 1.0
    eps_diff = (1 - max_eps) / (50 * hist_len * (1 - proportion_best))
    total_opt_traj = int(hist_len * 50 * proportion_best)
    n_opt_traj = 0

    while traj_counter < hist_len or total_opt_traj > n_opt_traj:
        buffer_loc = defaultdict(list)
        episode_starts = np.ones((num_envs,), dtype=bool)
        lstm_states = None

        done = False
        obs, _ = env.reset()
        ts = time.time()
        while not done:
            # test = env.np_random.random()
            test = np.random.uniform()

            with torch.no_grad():
                action_model, lstm_states = model.predict(
                    obs[0],
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
            lst = np.arange(8)[:action_model].tolist()
            lst.extend(np.arange(8)[action_model + 1 :].tolist())
            action_eps = np.random.choice(lst)

            action = action_eps if test < eps else action_model
            next_obs, reward, trunc, term, _ = env.step(action)

            buffer_loc["states"].append(copy.deepcopy(obs[0]))
            # buffer_loc["pos"].append(obs[1])
            buffer_loc["actions"].append(action)
            buffer_loc["rewards"].append(reward)
            # buffer_loc["goal"].append(goal)
            # buffer_loc["eps"].append(eps)

            done = trunc | term
            add_trajectory = reward > 0

            # if add_trajectory:
            #     steps_diff = 50 - len(buffer_loc["states"])
            # else:
            #     steps_diff = 1

            # eps = max(max_eps, eps - eps_diff * steps_diff)
            eps = max(max_eps, eps - eps_diff)
            if eps == max_eps:
                n_opt_traj += 1

            if add_trajectory or done:
                buffer_glob["states"].extend(buffer_loc["states"])
                buffer_glob["actions"].extend(buffer_loc["actions"])
                buffer_glob["rewards"].extend(buffer_loc["rewards"])
                # buffer_glob["goal"].extend(buffer_loc["goal"])
                # buffer_glob["pos"].extend(buffer_loc["pos"])
                # buffer_glob["eps"].extend(buffer_loc["eps"])

                traj_counter += 1

                break

            episode_starts = done
            obs = next_obs

        # print(f"GOAL {traj_counter} DONE IN: {time.time() - ts:.4f} sec")

    env.close()
    env = None
    model = None
    # ret_lst.append(buffer_glob)
    return buffer_glob, goal, os.getpid()


def killtree(pid, including_parent=False):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()

    if including_parent:
        parent.kill()


def create_model_env(model_path, goal):
    model = RecurrentPPO.load(f"{model_path}/best_model.zip", device="cuda")

    watermaze_setup = {
        "width": "96",
        "height": "72",
        "episodeLengthSeconds": "50",
        "fps": "1",
        "spawnRadius": "0",  # legacy
        "x": f"{goal[0]}",
        "y": f"{goal[1]}",
    }

    env = deepmind_lab.Lab(
        "contributed/dmlab30/rooms_watermaze",
        ["RGBD", "DEBUG.POS.TRANS"],
        config=watermaze_setup,
        renderer="hardware",
    )
    env = WrapObsMaze(
        WrapActionsMaze(shimmy.dm_lab_compatibility.DmLabCompatibilityV0(env))
    )

    return model, env


def write_to_file(file, buffer, i, goal):
    g = file.create_group(str(i))

    g.create_dataset("obs", data=np.stack(buffer["states"]), compression="gzip")
    g.create_dataset("action", data=np.stack(buffer["actions"]), compression="gzip")
    g.create_dataset("reward", data=np.stack(buffer["rewards"]), compression="gzip")
    # g.create_dataset("obs", data=np.stack(buffer["states"]))
    # g.create_dataset("action", data=np.stack(buffer["actions"]))
    # g.create_dataset("reward", data=np.stack(buffer["rewards"]))
    g.attrs["goal"] = goal


def filter_goals(goals, min_dist):
    goals = np.array(goals)
    mask = np.max(np.abs(goals), axis=1) >= min_dist
    return goals[mask].tolist(), mask


@pyrallis.wrap()
def generate_trajectory(config: GenConfig):
    # try:
    # except RuntimeError:
    #     pass

    wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
    )

    # np.random.seed(44)
    folders = os.listdir(config.model_weights)
    all_goals = [
        [float(g[0]), float(g[1])]
        for g in [f[1:-1].split(",") for f in folders if not f.startswith(".")]
    ]

    model_paths = np.array(
        [
            os.path.join(config.model_weights, f"({goal[0]},{goal[1]})")
            for goal in all_goals
        ]
    )

    goals, mask = filter_goals(all_goals, min_dist=300)
    model_paths = model_paths[mask]

    # goals, mask = filter_goals(goals, 200)
    tqdm.write(f"FILTERED GOALS NUM: {len(goals)}")

    if config.num_goals > len(goals):
        raise EnvironmentError("num_goals is greater than goals available")

    idx_goals = np.random.choice(
        np.arange(len(goals)), size=config.num_goals, replace=False
    )

    goals = [goals[i] for i in idx_goals]
    model_paths = [model_paths[i] for i in idx_goals]

    model_paths *= config.hist_per_goal
    goals *= config.hist_per_goal

    assert len(model_paths) == len(goals)

    save_path = os.path.join(
        config.save_path,
        f"{config.num_goals}g"
        f"-{config.hist_per_goal}hpg"
        f"-{config.hist_len}len"
        f"-{config.eps}eps"
        f"-{str(uuid.uuid4())[:4]}_prop:{config.proportion_best}",
    )

    tqdm.write(f"# CPU: {mp.cpu_count()}")
    tqdm.write(f"FILE NAME: {save_path}")
    tqdm.write(f"GOALS: {len(goals)}")

    os.makedirs(save_path, exist_ok=True)
    h5_file = h5py.File(f"{save_path}/data.hdf5", "w", track_order=True)

    pbar = tqdm(total=len(goals))
    g = 0

    snaps = []
    for i in range(0, len(goals), config.n_proc):
        with ProcessPoolExecutor(config.n_proc, mp_context=mp.get_context("spawn")) as e:
            # mp_context = mp.get_context("spawn")
            future_list = []
            for model, goal in zip(
                model_paths[i : i + config.n_proc], goals[i : i + config.n_proc]
            ):
                future = e.submit(
                    run_collecting,
                    (model, goal, i),
                    config.hist_len,
                    config.eps,
                    config.proportion_best,
                )
                future_list.append(future)

            for f in as_completed(future_list):
                buffer, goal, proc_pid = f.result()
                write_to_file(h5_file, buffer, g, goal)
                pbar.update(1)
                g += 1

                del buffer
                # del writers[str(goal)]
                killtree(proc_pid)

            gc.collect()

            # snaps.append(tracemalloc.take_snapshot())

    h5_file.close()
    print(f"Successfully collected {g} goals")

    try:
        with open(os.path.join(save_path, "train_goals.txt"), "w") as f:
            for g in goals:
                f.write(f"{g[0]},{g[1]}\n")
    except:
        print("saving goal list failed you dumbass")

    # [writer.finish() for writer in writers.values()]


if __name__ == "__main__":
    generate_trajectory()
