from acme.agents.jax.r2d2 import learning as r2d2_learning
import reverb
import time
from acme import types
from acme.jax import utils
import jax.numpy as jnp
import jax
from acme.wrappers import observation_action_reward
from acme.adders.reverb.base import Step
from random import randint
OAR = observation_action_reward.OAR

def reflect_observations(values: types.Nest) -> types.NestedArray:
  return jax.tree_map(
      lambda x: jnp.flip(x, axis=(3, 5)), values)

def reflect_actions(values: types.Nest) -> types.NestedArray:
  originals = [values == 1, values == 2, values == 4, values == 5]
  rflct = [2, 1, 5, 4]
  return jax.tree_map(
      lambda x: jnp.select(originals, rflct, default=x),values)

def reflect_samples(samples: reverb.ReplaySample) -> reverb.ReplaySample:
  """Reflects the observations in the samples."""
  reflected_observation = reflect_observations(samples.data.observation.observation)
  reflected_action = reflect_actions(samples.data.observation.action)
  reward = samples.data.observation.reward

  reflected_OAR = OAR(
      observation=reflected_observation,
      action=reflected_action,
      reward=reward)
  
  this_action = reflect_actions(samples.data.action)
  this_reward = samples.data.reward

  this_discount = samples.data.discount
  this_start_of_episode = samples.data.start_of_episode
  this_extras = samples.data.extras

  new_data = Step(
      observation=reflected_OAR,
      action=this_action,
      reward=this_reward,
      discount=this_discount,
      start_of_episode=this_start_of_episode,
      extras=this_extras)
  
  reflected_samples = reverb.ReplaySample(
      info=samples.info,
      data=new_data)
  
  return reflected_samples



class SimfishR2D2Learner(r2d2_learning.R2D2Learner):


    

  def step(self):
    prefetching_split = next(self._iterator)
    # The split_sample method passed to utils.sharded_prefetch specifies what
    # parts of the objects returned by the original iterator are kept in the
    # host and what parts are prefetched on-device.
    # In this case the host property of the prefetching split contains only the
    # replay keys and the device property is the prefetched full original
    # sample.
    keys = prefetching_split.host
    samples: reverb.ReplaySample = prefetching_split.device

    # toss a coin to decide whether regular or mirrored comes first
    if randint(0, 1) == 0:
      # Do a mirrored step first.
      mirrored = [True, False]
    else:
      # Do a regular step first.
      mirrored = [False, True]

    for mir in mirrored:
      start = time.time()
      if mir:
        self._state, priorities, metrics = self._sgd_step(
            self._state, reflect_samples(samples))
      else:
        self._state, priorities, metrics = self._sgd_step(self._state, samples)
      # Do a batch of SGD.
      # Take metrics from first replica.
      metrics = utils.get_from_first_device(metrics)
      # Update our counts and record it.
      counts = self._counter.increment(steps=1, time_elapsed=time.time() - start)

      # Update priorities in replay.
      if self._replay_client:
        self._async_priority_updater.put((keys, priorities))

      # Attempt to write logs.
      self._logger.write({**metrics, **counts})
