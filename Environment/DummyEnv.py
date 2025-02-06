import numpy as np
import dm_env
from dm_env import specs
from typing import NamedTuple
from acme import types
from acme.wrappers import observation_action_reward

OAR = observation_action_reward.OAR


class DummyEnv(dm_env.Environment):
    """
    A dummy environment that returns random observations.
    """
    def __init__(self):
        super().__init__()
        self._reset_next_step = True

    @staticmethod
    def get_random_observation():
        """
        Returns a random observation.
        """
        return np.random.rand(64, 64, 3).astype(np.float32)

    def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        return dm_env.restart(self.get_observation(action=0, reward=0.0))

    def step(self, action: int) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()

        done = np.random.rand() < 0.1
        observation = self.get_observation(action, 0.)
        if done:
            self._reset_next_step = True
            return dm_env.termination(reward=np.random.rand(), observation=observation)
        else:
            return dm_env.transition(reward=np.random.rand(), observation=observation)

    def observation_spec(self) -> specs.BoundedArray:
        obs_spec = specs.Array(shape=(64, 64, 3), dtype=np.float32)
        return OAR(
            observation=obs_spec,
            action=specs.Array(shape=(), dtype=int),
            reward=specs.Array(shape=(), dtype=np.float32),
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(dtype=int, num_values=2)

    def get_observation(self, action, reward):
        """
        Returns a random observation with the action and reward, in OAR format
        """
        return OAR(
            observation=self.get_random_observation(),
            action=action,
            reward=reward,
        )
