import warnings
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# gym warnings are annoying
warnings.filterwarnings("ignore")


class DarkKeyToDoor(gym.Env):
    metadata = {"render_mode": ["rgb_array"], "render_fps": 1}

    def __init__(
        self, size=9, goal_pos=None, render_mode=None, terminate_after_unlock=False
    ):
        self.size = size
        self.has_key = False

        # 0-th element: door position, 1-st element: key position
        if goal_pos is None:
            self.door_pos = self.generate_pos()
            self.key_pos = self.generate_pos()
        else:
            assert len(goal_pos) == 2, "goal pos defined incorrectly!"
            self.door_pos = goal_pos[0]
            self.key_pos = goal_pos[1]

        # TODO: check with (x,y) obs
        self.observation_space = spaces.Discrete(self.size**2)
        self.action_space = spaces.Discrete(5)

        self.action_to_direction = {
            0: np.array((0, 0), dtype=np.float32),  # noop
            1: np.array((-1, 0), dtype=np.float32),  # up
            2: np.array((0, 1), dtype=np.float32),  # right
            3: np.array((1, 0), dtype=np.float32),  # down
            4: np.array((0, -1), dtype=np.float32),  # left
        }
        self.center_pos = (self.size // 2, self.size // 2)
        # Flag: True if the key is collected
        self.terminate_after_unlock = terminate_after_unlock
        self.render_mode = render_mode

    def generate_pos(self):
        # Generate random dot in the grid of type float32
        return np.random.randint(0, self.size, size=2).astype(np.float32)

    def pos_to_state(self, pos):
        return int(pos[0] * self.size + pos[1])

    def state_to_pos(self, state):
        return np.array(divmod(state, self.size))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.agent_pos = self.generate_pos()
        self.has_key = False

        return self.pos_to_state(self.agent_pos), {}

    def step(self, action):
        self.agent_pos = np.clip(
            self.agent_pos + self.action_to_direction[action], 0, self.size - 1
        )

        key_reward = (
            1.0
            if np.array_equal(self.agent_pos, self.key_pos) and not self.has_key
            else 0.0
        )

        if key_reward != 0.0:
            self.has_key = True

        door_reward = (
            1.0
            if np.array_equal(self.agent_pos, self.door_pos) and self.has_key
            else 0.0
        )

        terminated = True if door_reward and self.terminate_after_unlock else False

        return (
            self.pos_to_state(self.agent_pos),
            key_reward + door_reward,
            terminated,
            False,
            {},
        )

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            # Create a grid representing the dark room
            grid = np.full(
                (self.size, self.size, 3), fill_value=(255, 255, 255), dtype=np.uint8
            )
            grid[self.key_pos[0], self.key_pos[1]] = (255, 255, 0)
            grid[self.door_pos[0], self.door_pos[1]] = (128, 64, 48)
            grid[int(self.agent_pos[0]), int(self.agent_pos[1])] = (0, 255, 0)
            return grid
