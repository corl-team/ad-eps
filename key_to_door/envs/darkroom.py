import warnings
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# gym warnings are annoying
warnings.filterwarnings("ignore")


class DarkRoom(gym.Env):
    metadata = {"render_mode": ["rgb_array"], "render_fps": 1}

    def __init__(self, size=9, goal_pos=None, render_mode=None, terminate_on_goal=False):
        self.size = size
        # TODO: check with (x,y) obs
        self.observation_space = spaces.Discrete(self.size**2)
        self.action_space = spaces.Discrete(5)

        self.action_to_direction = {
            0: np.array((0, 0), dtype=np.float32),   # noop
            1: np.array((-1, 0), dtype=np.float32),  # up
            2: np.array((0, 1), dtype=np.float32),   # right
            3: np.array((1, 0), dtype=np.float32),   # down
            4: np.array((0, -1), dtype=np.float32),  # left
        }
        self.center_pos = (self.size // 2, self.size // 2)
        if goal_pos is not None:
            self.goal_pos = np.asarray(goal_pos)
            assert self.goal_pos.ndim == 1
        else:
            self.goal_pos = self.generate_goal_pos()

        self.terminate_on_goal = terminate_on_goal
        self.render_mode = render_mode

    def generate_goal_pos(self):
        return self.np_random.integers(0, self.size, size=2)

    def pos_to_state(self, pos):
        return int(pos[0] * self.size + pos[1])

    def state_to_pos(self, state):
        return np.array(divmod(state, self.size))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.agent_pos = np.array(self.center_pos, dtype=np.float32)

        return self.pos_to_state(self.agent_pos), {}

    def step(self, action):
        self.agent_pos = np.clip(self.agent_pos + self.action_to_direction[action], 0, self.size - 1)

        reward = 1.0 if np.array_equal(self.agent_pos, self.goal_pos) else 0.0
        terminated = True if reward and self.terminate_on_goal else False

        return self.pos_to_state(self.agent_pos), reward, terminated, False, {}

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            # Create a grid representing the dark room
            grid = np.full((self.size, self.size, 3), fill_value=(255, 255, 255), dtype=np.uint8)
            grid[self.goal_pos[0], self.goal_pos[1]] = (255, 0, 0)
            grid[int(self.agent_pos[0]), int(self.agent_pos[1])] = (0, 255, 0)
            return grid