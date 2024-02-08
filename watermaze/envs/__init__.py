import gymnasium as gym
from functools import partial

from .darkroom import DarkRoom
from .dark_key_to_door import DarkKeyToDoor


def _make_env_fn(*args, **kwargs):
    return DarkRoom(*args, **kwargs)


def _make_env_fn_k2d(*args, **kwargs):
    return DarkKeyToDoor(*args, **kwargs)


# Register the environments with Gym
gym.envs.register(
    id="DarkRoomSmall-v0",
    entry_point=partial(_make_env_fn, size=3, terminate_on_goal=False),
    max_episode_steps=5,
)

gym.envs.register(
    id="DarkRoomMedium-v0",
    entry_point=partial(_make_env_fn, size=5, terminate_on_goal=False),
    max_episode_steps=20,
)

gym.envs.register(
    id="DarkRoom-v0",
    entry_point=partial(_make_env_fn, size=9, terminate_on_goal=False),
    max_episode_steps=20,
)

gym.envs.register(
    id="DarkRoomHard-v0",
    entry_point=partial(_make_env_fn, size=17, terminate_on_goal=True),
    max_episode_steps=20,
)

gym.envs.register(
    id="DarkKey2Door-v0",
    entry_point=partial(
        _make_env_fn_k2d, size=9, terminate_after_unlock=True, max_episode_steps=40
    ),
    max_episode_steps=40,
)

gym.envs.register(
    id="DarkKey2DoorSmall-v0",
    entry_point=partial(
        _make_env_fn_k2d, size=3, terminate_after_unlock=True, max_episode_steps=5
    ),
    max_episode_steps=5,
)

# I don't know why but gymnasium code owners have decided not to include 'max_episode_steps' into env's attributes
# Somebody has invented a wrapper around an env to inject this info, but it makes code very clumsy,
# so it is easier just to provide this attribute manually *cringe*
