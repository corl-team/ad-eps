import uuid

import gymnasium as gym

from generate_trajectories import generate_data
from utils.data import SequenceDataset
from utils.visualization import animate_traj

if __name__ == '__main__':
    savedir = f'trajectories-{str(uuid.uuid4())[:2]}'
    max_eps = 0.0
    train_goals, test_goals = generate_data(env_name='DarkKey2Door-v0',
                                            savedir=savedir,
                                            num_train_goals=5,
                                            hist_per_goal=100,
                                            max_eps=max_eps,
                                            seed=1)

    dataset = SequenceDataset(
        runs_path=savedir,
        seq_len=160,
        subsample=1
    )

    states = dataset._states[-40:]
    actions = dataset._actions[-40:]
    goal = train_goals[-1]

    env = gym.make('DarkKey2Door-v0')

    animate_traj(env, states, actions, name=f'max_perf_{max_eps}', key_pos=goal[0], goal=goal[1])

