import json
import multiprocessing as mp
import os
import uuid
from collections import defaultdict
from functools import partial

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from algorithms.optimal_agent import OptimalAgent
from utils.misc import set_seed, train_test_goals


# @dataclass
# class Config:
#     seed: int = 0
#     savedir: str = 'trajectories/'
#     num_train_goals: int = 80
#     env_name: str = 'DarkRoom-v0'
#     hist_per_goal: int = 5000
#     max_perf: float = 1.0


def generate_trajectory(
    train_goal,
    actor,
    env_name,
    savedir,
    n_episodes=25_000,
    debug=False,
    max_perf=0.0,
    seed=0,
):
    env = gym.make(env_name, goal_pos=train_goal)
    episodes = defaultdict(list)
    episode_rewards = []

    eps = 1.0
    eps_diff = 1.0 / (20 * n_episodes * 0.9)  # TODO: remove hardcoded 20

    for i in range(n_episodes):
        trajectories = defaultdict(list)

        (state, _), done = env.reset(), False
        episode_reward = 0.0
        while not done:
            if np.random.uniform() < eps:
                action = env.action_space.sample()
            else:
                action = actor.act(env.state_to_pos(state), env)

            new_state, reward, term, trunc, _ = env.step(action)
            episode_reward += reward
            done = term | trunc

            trajectories["states"].append(state)
            trajectories["actions"].append(action)
            trajectories["rewards"].append(reward)
            trajectories["terminateds"].append(term)
            trajectories["truncateds"].append(trunc)
            trajectories["eps"].append(eps)

            state = new_state
            eps = max((1 - max_perf), eps - eps_diff)

        if debug and i % 500 == 0:
            tqdm.write(
                f" EPS: {eps:.3f}, last 10 returns: {np.mean(episode_rewards[-10:])}"
            )

        episodes["states"].append(trajectories["states"])
        episodes["actions"].append(trajectories["actions"])
        episodes["rewards"].append(trajectories["rewards"])
        episodes["terminateds"].append(trajectories["terminateds"])
        episodes["truncateds"].append(trajectories["truncateds"])
        episodes["eps"].append(trajectories["eps"])

        episode_rewards.append(episode_reward)

    if debug:
        tqdm.write(
            f" EPS: {eps:.3f}, last 10 returns: {np.mean(episode_rewards[-10:])}"
        )

    idd = str(uuid.uuid4())[:3]
    filename = dump_trajectories(savedir, idd, episodes)

    return filename


# @pyrallis.wrap()
def generate_data(env_name, savedir, num_train_goals, hist_per_goal, max_perf, seed):
    set_seed(seed)
    actor = OptimalAgent()
    os.makedirs(savedir, exist_ok=True)
    savedir = os.path.join(savedir, str(uuid.uuid4())[:5])
    os.makedirs(savedir, exist_ok=True)

    train_goals, test_goals = train_test_goals(
        grid_size=gym.make(env_name).unwrapped.size,
        num_train_goals=num_train_goals,
        seed=seed,
    )

    print(f"\nGENERATING {hist_per_goal} EPISODES PER GOAL")
    generate_partial = partial(
        generate_trajectory,
        actor=actor,
        env_name=env_name,
        savedir=savedir,
        n_episodes=hist_per_goal,
        max_perf=max_perf,
        debug=False,
    )
    with mp.Pool(processes=os.cpu_count()) as pool:
        # with mp.Pool(processes=1) as pool:
        filenames = pool.map(generate_partial, train_goals.tolist())

    save_metadata(savedir, train_goals, filenames, alg_name="opt-eps")

    return train_goals, test_goals


def dump_trajectories(savedir, std, trajectories):
    filename = os.path.join(savedir, f"trajectories_{std}.npz")
    np.savez(
        filename,
        states=np.array(trajectories["states"], dtype=float),
        actions=np.array(trajectories["actions"]),
        rewards=np.array(trajectories["rewards"], dtype=float),
        dones=np.int32(
            np.array(trajectories["terminateds"]) | np.array(trajectories["truncateds"])
        ),
        eps=np.array(trajectories["eps"])
    )

    return os.path.basename(filename)


def save_metadata(savedir, goal_pos, save_filenames, alg_name="Q-learning"):
    if type(goal_pos) is np.ndarray:
        goal_pos = goal_pos.tolist()
    metadata = {
        "algorithm": f"{alg_name}",
        "label": "label",
        "ordered_trajectories": save_filenames,
        "goal": goal_pos,
    }
    with open(os.path.join(savedir, "metadata.metadata"), mode="w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    generate_data()
