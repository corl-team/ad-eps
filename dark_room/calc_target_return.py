import os
from multiprocessing import Pool

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from algorithms.optimal_agent import OptimalAgent


# @dataclass
# class Config:
#     seed: int = 0
#     savedir: str = 'trajectories/'
#     num_train_goals: int = 80
#     env_name: str = 'DarkRoom-v0'
#     hist_per_goal: int = 5000
#     max_perf: float = 1.0


def generate_trajectory(train_goals, eps, actor, env_name, n_episodes=25_000):
    episode_rewards = []

    env = gym.make(env_name, goal_pos=train_goals)
    env.reset(seed=0)

    states = []
    actions = []

    rand = 0
    opt = 0

    for i in range(n_episodes):
        (state, _), done = env.reset(), False
        episode_reward = 0.0
        while not done:
            if np.random.uniform() < eps:
                action = env.action_space.sample()
                rand += 1
            else:
                action = actor.act(env.state_to_pos(state), env)
                opt += 1

            new_state, reward, term, trunc, _ = env.step(action)
            episode_reward += reward
            done = term | trunc

            states.append(state)
            actions.append(action)

            state = new_state

        episode_rewards.append(episode_reward)

    # print(f" EPS: {eps:.3f}, Returns: {np.mean(episode_rewards)}")

    return episode_rewards


# @pyrallis.wrap()
def calculate_eps_reward(goals, env_name, eps, n_episodes=1000):
    actor = OptimalAgent()

    gen_traj_partial = partial(
        generate_trajectory,
        eps=eps,
        actor=actor,
        env_name=env_name,
        n_episodes=n_episodes,
    )

    with Pool(os.cpu_count()) as p:
        r = list(tqdm(p.imap(gen_traj_partial, goals), total=len(goals)))

    rets = np.concatenate(r)
    return rets.mean()
