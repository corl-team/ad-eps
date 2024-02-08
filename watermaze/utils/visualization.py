import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from matplotlib import animation
from PIL import Image
import logging
import matplotlib

matplotlib.use("Agg")  # Set the backend before importing pyplot
import matplotlib.pyplot as plt

logging.getLogger("matplotlib").setLevel(logging.ERROR)


def draw_grid(grid_size):
    fig, ax = plt.subplots(figsize=(grid_size, grid_size))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    # Major ticks
    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))
    # Labels for major ticks
    ax.set_xticklabels(np.arange(0, grid_size, 1), minor=True)
    ax.set_yticklabels(np.arange(0, grid_size, 1), minor=True)

    # Minor ticks
    ax.set_xticks(np.arange(0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(0.5, grid_size, 1), minor=True)
    ax.tick_params(
        which="major", bottom=False, left=False, labelbottom=False, labelleft=False
    )

    ax.grid(which="major", axis="both", color="k")
    colormesh = ax.pcolormesh(np.zeros((grid_size, grid_size)))

    return fig, ax, colormesh


def animate_traj(env, states, actions, goal, key_pos=None, name=""):
    size = env.size
    fig, ax, colormesh = draw_grid(size)
    states = env.state_to_pos(states)

    actions_dict = {0: "O", 1: "↓", 2: "→", 3: "↑", 4: "←"}

    def animate(i, goal):
        mesh = np.zeros((size, size))
        mesh[int(states[0][i]), int(states[1][i])] = 1
        # mesh[np.random.randint(0, size), np.random.randint(0, size)] = 1
        mesh[goal[0], goal[1]] = 3

        if key_pos is not None:
            mesh[key_pos[0], key_pos[1]] = 2
            ax.text(key_pos[1] + 0.6, key_pos[0] + 0.6, "¶", color="red", fontsize=15)

        colormesh = ax.pcolormesh(mesh)

        act = actions_dict[actions[i]]
        ax.text(states[1][i] + 0.4, states[0][i] + 0.4, act, color="orange", fontsize=20)

        return colormesh

    animate = partial(animate, goal=goal)

    anim = animation.FuncAnimation(fig, animate, frames=states.shape[1], blit=False)
    anim.save(f"basic_animation_{name}.gif", fps=2)


def animate_watermaze(obs_hist):
    images = []
    for obs in obs_hist:
        im = Image.fromarray(obs.transpose(1, 2, 0))
        # im = obs
        images.append(im)

    images[0].save(
        "watermaze_traj.gif",
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=200,
        loop=0,
    )


def draw_sample_eff_graph(
    eval_res, name, ylim=None, max_return=None, max_return_eps=None
):
    rets = np.vstack([h for h in eval_res.values()])
    means = rets.mean(0)
    stds = rets.std(0)
    x = np.arange(1, rets.shape[1] + 1)

    fig, ax = plt.subplots(dpi=100)
    ax.grid(visible=True)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.plot(x, means)
    ax.fill_between(x, means - stds, means + stds, alpha=0.2)

    ax.set_ylabel("Return")
    ax.set_xlabel("Episodes In-Context")
    ax.set_title(f"{name}")

    if max_return is not None:
        ax.axhline(
            max_return,
            ls="--",
            color="goldenrod",
            lw=2,
            label=f"optimal_return: {max_return:.2f}",
        )
    if max_return_eps is not None:
        ax.axhline(
            max_return_eps,
            ls="--",
            color="indigo",
            lw=2,
            label=f"max_perf_return: {max_return_eps:.2f}",
        )
    if max_return_eps is not None or max_return is not None:
        plt.legend()

    fig.savefig(f"rets_vs_eps_{name}.png")

    return f"rets_vs_eps_{name}.png"