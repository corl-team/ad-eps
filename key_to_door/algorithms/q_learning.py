import json
import os
import random
from collections import defaultdict

import numpy as np


def eval(env, Q, only_optimal=False):
    eval_trajectories = defaultdict(list)
    state, _ = env.reset()
    while True:
        a = np.argmax(Q[state,:])
        new_state, r, term, trunc, _ = env.step(a)

        eval_trajectories['states'].append(state)
        eval_trajectories['actions'].append(a)
        eval_trajectories['rewards'].append(r)
        eval_trajectories['terminateds'].append(term)
        eval_trajectories['truncateds'].append(trunc)

        state = new_state

        if term or trunc:
            break

    if only_optimal:
        if eval_trajectories['rewards'][-1] == 1:
            return eval_trajectories
        else:
            return {}
    else:
        return eval_trajectories


def dump_trajectories(savedir, i, trajectories):
    filename = os.path.join(savedir, f'trajectories_{i}.npz')
    np.savez(
        filename,
        states=np.array(trajectories['states'], dtype=float).reshape(-1, 1),
        actions=np.array(trajectories['actions']).reshape(-1, 1),
        rewards=np.array(trajectories['rewards'], dtype=float).reshape(-1, 1),
        dones=np.int32(np.array(trajectories['terminateds']) | np.array(trajectories['truncateds'])).reshape(-1, 1),
        # qtables=np.array(trajectories['qtables']),
    )

    return os.path.basename(filename)


def save_metadata(savedir, goal_pos, save_filenames):
    metadata = {
        "algorithm": "Q-learning",
        "label": "label",
        "ordered_trajectories": save_filenames,
        "goal": goal_pos.tolist()
    }
    with open(os.path.join(savedir, 'metadata.metadata'), mode="w") as f:
        json.dump(metadata, f, indent=2)


def q_learning(env, lr=0.01, discount=0.9, num_steps=int(1e7), save_every=1000, savedir='tmp', only_optimal=False, seed=None):
    # Q = np.random.randn(env.size * env.size, env.action_space.n)
    Q = np.zeros(shape=(env.size * env.size, env.action_space.n))

    state, _ = env.reset(seed=seed)
    trajectories = defaultdict(list)
    eval_trajectories = defaultdict(list)
    save_filenames = []
    eval_save_filenames = []
    #creating lists to contain total rewards and steps per episode
    eps = 1.
    eps_diff = 1.8 / num_steps
    term, trunc = False, False
    for i in range(1, num_steps + 1):
        if term or trunc:
            # Get trajectories with optimal actions

            state, _ = env.reset()

        if random.random() < eps:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[state, :])

        next_state, r, term, trunc, _ = env.step(a)

        if term:
            Q[next_state, :] = 0

        # Collect trajectories with exploratory actions
        trajectories['states'].append(state)
        trajectories['actions'].append(a)
        trajectories['rewards'].append(r)
        trajectories['terminateds'].append(term)
        trajectories['truncateds'].append(trunc)
        trajectories['qtables'].append(Q)

        #Update Q-Table with new knowledge
        Q[state, a] += lr * (r + discount * np.max(Q[next_state, :]) - Q[state, a])
        state = next_state

        eps = max(0, eps - eps_diff)

        # dump training trajectories
        if i % save_every == 0:
            filename = dump_trajectories(savedir, i, trajectories)
            save_filenames.append(os.path.basename(filename))
            trajectories = defaultdict(list)

            if len(eval_trajectories['states']) != 0:
                eval_filename = dump_trajectories('eval_' + savedir, i, eval_trajectories)
                eval_save_filenames.append(os.path.basename(eval_filename))
                eval_trajectories = defaultdict(list)

    save_metadata(savedir, env.goal_pos, save_filenames)
    save_metadata('eval_' + savedir, env.goal_pos, eval_save_filenames)

    return Q