import torch
from torchvision import transforms as T
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from utils import GREEN, RESET


class SkipFrame(gym.Wrapper):
    """Return only every `skip`-th frame."""

    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        """Repeat action, and sum reward."""
        total_reward = 0.0
        for i in range(self.skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert RGB observation to grayscale."""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    """Resize (downsize) and normalize observation. Originally (240, 256)."""

    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


def make_env(name, render_mode, shape=84, skip=4, moveset=[["right"], ["right", "A"]]):
    """
    Creates and configures a Super Mario Bros environment.

    Args:
        name (str): The name of the Super Mario Bros environment to create.
                    Valid names can be found at https://github.com/Kautenja/gym-super-mario-bros?tab=readme-ov-file#environments
        render_mode (str): The rendering mode for the environment. Choices are 'human' and 'rgb_array'.
        shape (int or tuple): The shape of the resized frame.
        skip (int): The number of frames to stack into one.
        moveset (list): The moveset to use for the environment. See gym_super_mario_bros.actions.

    Returns:
        gym.Env: The configured Super Mario Bros environment.
    """
    # Create the environment
    env = gym_super_mario_bros.make(
        name, render_mode=render_mode, apply_api_compatibility=True
    )
    env.moveset = moveset
    env.name = name

    # Apply wrappers to env - https://gymnasium.farama.org/api/wrappers/
    env = JoypadSpace(env, moveset)
    env = SkipFrame(env, skip=skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=shape)
    env = FrameStack(env, num_stack=skip)
    env.reset()

    # Logging
    params = dict(
        name=name,
        render_mode=render_mode,
        shape=env.shape,
        skip=env.skip,
        moveset=env.moveset,
    )
    print(
        f"{GREEN}Initialized environment with the following params:{RESET}\n{params}\n"
    )

    return env


if __name__ == "__main__":
    """
    Prints information on return values from env.step() and runs a random agent for a few seconds.
    """

    name = "SuperMarioBrosRandomStages-v0"
    render_mode = "human"
    steps = 250

    env = make_env(name=name, render_mode=render_mode)
    next_state, reward, done, trunc, info = env.step(action=0)
    print(
        f"State Shape: {next_state.shape},\nReward: {reward},\nDone: {done},\nInfo: {info}"
    )

    try:
        state = env.reset()
        while True:
            next_state, reward, done, trunc, info = env.step(env.action_space.sample())
            if done or info["flag_get"]:
                print("Game over.")
                break
    except KeyboardInterrupt:
        print("Game stopped by user.")
    finally:
        env.close()
