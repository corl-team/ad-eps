import uuid

import gymnasium as gym

from generate_trajectories import generate_data
from utils.data import SequenceDataset
from utils.visualization import animate_traj

if __name__ == '__main__':
    savedir = f'trajectories-{str(uuid.uuid4())[:2]}'
    max_perf = 0.7
    train_goals, test_goals = generate_data(env_name='DarkRoom-v0',
                                            savedir=savedir,
                                            num_train_goals=1,
                                            hist_per_goal=100,
                                            max_perf=max_perf,
                                            seed=1)

    dataset = SequenceDataset(
        runs_path=savedir,
        seq_len=80,
        subsample=1
    )

    states = dataset._states[-20:]
    actions = dataset._actions[-20:]
    goal = train_goals[-1]

    env = gym.make('DarkRoom-v0')

    animate_traj(env, states, actions, goal, name=f'max_perf_{max_perf}')

