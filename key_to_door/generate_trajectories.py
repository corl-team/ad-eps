import json
import multiprocessing as mp
import os
import uuid
from collections import defaultdict
from functools import partial

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from algorithms.optimal_agent import OptimalAgentK2D
from utils.misc import train_test_goals_k2d


# @dataclass
# class Config:
#     seed: int = 0
#     savedir: str = 'trajectories/'
#     num_train_goals: int = 80
#     env_name: str = 'DarkRoom-v0'
#     hist_per_goal: int = 5000
#     это : float = 1.0


def pad_to_maxlen(traj, maxlen):
    zeros = np.zeros(maxlen - len(traj["states"]), dtype=int).tolist()

    mask = np.hstack(
        [np.ones(len(traj["states"])), np.zeros(maxlen - len(traj["states"]))]
    )

    for k in traj.keys():
        if k in ["states", "actions", "rewards"]:
            traj[k].extend(zeros)

    return traj, mask


def generate_trajectory(
    env, actor, savedir, n_episodes=25_000, debug=False, max_eps=0.0
):
    episodes = defaultdict(list)
    episode_rewards = []

    eps = 1.0
    eps_diff = (1 - max_eps) / (env.max_steps * n_episodes * 0.9)

    for i in range(n_episodes):
        trajectories = defaultdict(list)

        (state, _), done = env.reset(), False
        episode_reward = 0.0
        while not done:
            test = env.np_random.random()
            if test < eps:
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

            if done:
                steps_diff = env.max_steps - len(trajectories['states'])
            else:
                steps_diff = 1

            eps = max(max_eps, eps - eps_diff * steps_diff)

        # trajectories, mask = pad_to_maxlen(trajectories, env.max_steps)

        episodes["states"].append(trajectories["states"])
        episodes["actions"].append(trajectories["actions"])
        episodes["rewards"].append(trajectories["rewards"])
        episodes["terminateds"].append(trajectories["terminateds"])
        episodes["truncateds"].append(trajectories["truncateds"])
        episodes["eps"].append(trajectories["eps"])
        # episodes["mask"].append(mask)

        episode_rewards.append(episode_reward)

    # if debug:
    #     tqdm.write(
    #         f" GENERATED MAX EPS: {eps:.3f}, last 10 returns: {np.mean(episode_rewards[-10:])}"
    #     )

    idd = str(uuid.uuid4())[:5]
    filename = dump_trajectories(savedir, idd, episodes)

    return filename, episode_rewards[-10:], eps


def generate_envs(goals, env_name, seed):
    envs = []

    for i, goal in enumerate(goals):
        env = gym.make(env_name, goal_pos=goal)
        env.reset(seed=seed + i)
        env.action_space.seed(seed + i)
        envs.append(env)

    return envs


# @pyrallis.wrap()
def generate_data(
    env_name,
    savedir,
    num_train_goals,
    hist_per_goal,
    max_eps,
    seed,
    num_test_goals=None,
):
    actor = OptimalAgentK2D()
    os.makedirs(savedir, exist_ok=True)
    savedir = os.path.join(savedir, str(uuid.uuid4())[:5])
    os.makedirs(savedir, exist_ok=True)

    train_goals, test_goals = train_test_goals_k2d(
        grid_size=gym.make(env_name).unwrapped.size,
        num_train_goals=num_train_goals,
        seed=seed,
        num_test_goals=num_test_goals,
    )

    envs = generate_envs(train_goals, env_name, seed=seed)

    print(f"\nGENERATING {hist_per_goal} EPISODES PER GOAL")
    generate_partial = partial(
        generate_trajectory,
        actor=actor,
        savedir=savedir,
        n_episodes=hist_per_goal,
        max_eps=max_eps,
        debug=True,
    )

    with mp.Pool(processes=os.cpu_count()) as pool:
    # with mp.Pool(processes=1) as pool:
        rets = pool.map(generate_partial, envs)

    filenames = [r[0] for r in rets]
    last_ten_rets = [r[1] for r in rets]

    tqdm.write(f" LAST 10 RETURNS: {np.mean(last_ten_rets)}")

    save_metadata(savedir, train_goals, filenames, alg_name="opt-eps")

    return train_goals, test_goals


def dump_trajectories(savedir, std, trajectories):
    filename = os.path.join(savedir, f"trajectories_{std}.npz")
    np.savez(
        filename,
        states=trajectories["states"],
        actions=trajectories["actions"],
        rewards=trajectories["rewards"],
        # dones=np.int32(
        #     np.array(trajectories["terminateds"]) | np.array(trajectories["truncateds"])
        # ),
        # eps=np.array(trajectories["eps"]),
        # mask=np.array(trajectories["mask"]),
    )
    # np.save(filename, trajectories)
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
