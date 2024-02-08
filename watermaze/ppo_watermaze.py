import gymnasium as gym

gym.logger.set_level(40)

import deepmind_lab
import shimmy
import numpy as np
import torch.cuda
import torchvision

torchvision.disable_beta_transforms_warning()

from functools import partial
from sb3_contrib import RecurrentPPO
from typing import Optional
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.spaces import Discrete
import uuid
from utils.visualization import animate_watermaze
import wandb

from dataclasses import asdict, dataclass
import os
import pyrallis

from utils.misc import set_seed

from utils.data import generate_goals

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    # wandb params
    project: str = "better-than"
    group: str = "ppo-watermaze-oracle-new"
    name: str = "LSTM-PPO"
    # model params
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # training params
    n_steps: int = 50
    epochs: int = 10
    learning_rate: float = 3e-4
    batch_size: int = 8
    num_updates: int = 10_000_000
    # evaluation params
    eval_every: int = 1
    # general params
    checkpoints_path: Optional[str] = None
    train_seed: int = 10
    eval_seed: int = 42
    save_path: str = "demonstrator/"
    # watermaze
    episode_length_seconds: int = 50
    fps: int = 1
    # goal: Tuple[float, float] = (50.0, 120.0)
    goal_x: float = 100
    goal_y: float = 100

    def __post_init__(self):
        goal = generate_goals(200, 1, generator=np.random.default_rng())

        self.goal_x = np.round(goal[0][0], 3)
        self.goal_y = np.round(goal[0][1], 3)

        self.name = (
            f"{self.name}" f"-{str(uuid.uuid4())[:8]}" f"-({self.goal_x}, {self.goal_y})"
        )
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class WrapObsMaze(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            shape=(3, 72, 96), low=0, high=255, dtype=np.uint8
        )

    def observation(self, obs):
        return obs["RGBD"][:3]


class WrapActionsMaze(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(8)

        self.map_actions = {
            0: np.array([0, 0, 2]),  # forward
            1: np.array([0, 0, 1]),  # backward
            2: np.array([0, 1, 0]),  # left
            3: np.array([0, 2, 0]),  # right
            4: np.array([1, 0, 0]),  # look left
            5: np.array([2, 0, 0]),  # look right
            6: np.array([1, 0, 2]),  # forward and left
            7: np.array([2, 0, 2]),  # forward and right
        }

        self.look_l_r = np.array([0, -12.0, 12.0], dtype=np.float64)
        self.back_forward = np.array([0, -1, 1], dtype=np.float64)
        self.strafe_l_r = np.array([0, -1, 1], dtype=np.float64)

    def action(self, act_num):
        if isinstance(act_num, np.ndarray):
            assert act_num.ndim == 0
            act_num = act_num.item()
        act = self.map_actions[act_num]
        return {
            "CROUCH": np.array([self.look_l_r[act[0]]]),  # look right: 1 left: -1
            "FIRE": np.array([0.0]),  # look down: 1 up: -1; garbage
            "JUMP": np.array([self.strafe_l_r[act[1]]]),  # strafe left-right -1..1
            "LOOK_DOWN_UP_PIXELS_PER_FRAME": np.array(
                [self.back_forward[act[2]]]
            ),  # move forward: 1 back: -1
            "LOOK_LEFT_RIGHT_PIXELS_PER_FRAME": np.array([0.0]),  # garbage
            "MOVE_BACK_FORWARD": np.array([0.0]),  # garbage
            "STRAFE_LEFT_RIGHT": np.array([0.0]),  # garbage
        }


def create_envs(ep_len_sec, fps, goal, seed=None, n_envs=None):
    if n_envs is None:
        n_envs = os.cpu_count()

    x_goal, y_goal = goal

    def _init():
        watermaze_setup = {
            "width": "96",
            "height": "72",
            "episodeLengthSeconds": str(ep_len_sec),
            "fps": str(fps),
            "spawnRadius": "0",  # legacy
            "x": str(x_goal),
            "y": str(y_goal),
        }

        env = deepmind_lab.Lab(
            "contributed/dmlab30/rooms_watermaze",
            ["RGBD"],
            config=watermaze_setup,
            renderer="hardware",
        )
        env = WrapObsMaze(
            WrapActionsMaze(shimmy.dm_lab_compatibility.DmLabCompatibilityV0(env))
        )
        return env

    vec_env = make_vec_env(
        _init,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=partial(SubprocVecEnv, start_method="spawn"),
    )
    return vec_env


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed)

    vec_env = create_envs(
        config.episode_length_seconds,
        config.fps,
        [config.goal_x, config.goal_y],
        config.train_seed,
    )
    test_env = create_envs(
        config.episode_length_seconds,
        config.fps,
        goal=[config.goal_x, config.goal_y],
        n_envs=1,
        seed=config.eval_seed,
    )
    print(f"\n N_ENVS: {vec_env.num_envs}")

    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )

    model = RecurrentPPO(
        "CnnLstmPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        n_epochs=config.epochs,
        n_steps=config.n_steps,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        normalize_advantage=config.normalize_advantage,
        vf_coef=config.vf_coef,
        ent_coef=config.ent_coef,
        max_grad_norm=config.max_grad_norm,
        seed=config.train_seed,
        device=DEVICE,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
    )

    model.learn(
        total_timesteps=150_000,
        log_interval=config.eval_every,
        reset_num_timesteps=False,
    )

    obs_history, returns = evaluate(model, test_env, num_envs=1)
    animate_watermaze(obs_history)
    wandb.log(
        {
            "rollout/video": wandb.Video("watermaze_traj.gif"),
            "rollout/test_return": np.mean(returns),
        }
    )

    print("training is over...")
    run.finish()
    vec_env.close()
    test_env.close()


def evaluate(model, vec_env, num_envs):
    obs_history = []
    # cell and hidden state of the LSTM
    returns = []
    for i in range(30):
        obs = vec_env.reset()
        episode_starts = np.ones((num_envs,), dtype=bool)
        lstm_states = None
        done = False
        while not done:
            obs_history.append(obs[0])

            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            obs, rewards, trunc, term = vec_env.step(action)
            done = trunc | term[0]["TimeLimit.truncated"]
            episode_starts = done

            if rewards[0] > 0:
                returns.append(1)
                break

            if done:
                returns.append(0)

    return obs_history, returns


if __name__ == "__main__":
    train()
