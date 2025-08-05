from acme.agents.jax.r2d2 import learning as r2d2_learning
import reverb
import time
from acme import types
from acme.jax import utils
import jax.numpy as jnp
import jax
from acme.wrappers import observation_action_reward
from acme.adders.reverb.base import Step
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple
from acme.jax import networks as networks_lib
from acme.agents.jax.r2d2 import networks as r2d2_networks
import optax
import rlax
from acme.utils import counting
from acme.utils import loggers

OAR = observation_action_reward.OAR
R2D2ReplaySample = utils.PrefetchingSplit

def reflect_observations(values: types.Nest) -> types.NestedArray:
  return jax.tree_map(
      lambda x: jnp.flip(x, axis=(3, 5)), values)

def reflect_actions(values: types.Nest, actions_mirror: Dict[int, int]) -> types.NestedArray:
  originals = [values == ori for ori in actions_mirror.keys()]
  #originals = [values == 1, values == 2, values == 4, values == 5, values == 7, values == 8, values == 10, values == 11]
  rflct = [actions_mirror[ori] for ori in actions_mirror.keys()]
  #rflct = [2, 1, 5, 4, 8, 7, 11, 10]
  return jax.tree_map(
      lambda x: jnp.select(originals, rflct, default=x),values)

def reflect_samples(samples: reverb.ReplaySample, actions_mirror: Dict[int, int]) -> reverb.ReplaySample:
  """Reflects the observations in the samples."""
  reflected_vis_observation = reflect_observations(samples.data.observation.observation[0])
  internal_states = samples.data.observation.observation[1]
  reflected_action = reflect_actions(samples.data.observation.action, actions_mirror)
  reward = samples.data.observation.reward

  reflected_OAR = OAR(
      observation=[reflected_vis_observation, internal_states],
      action=reflected_action,
      reward=reward)
  
  this_action = reflect_actions(samples.data.action, actions_mirror)
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
  """A learner for the Simfish R2D2 agent, with support for reflection."""
  def __init__(self,
               networks: r2d2_networks.R2D2Networks,
               batch_size: int,
               random_key: networks_lib.PRNGKey,
               burn_in_length: int,
               discount: float,
               importance_sampling_exponent: float,
               max_priority_weight: float,
               target_update_period: int,
               iterator: Iterator[R2D2ReplaySample],
               optimizer: optax.GradientTransformation,
               actions_mirror: Dict[int, int],
               bootstrap_n: int = 5,
               tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR,
               clip_rewards: bool = False,
               max_abs_reward: float = 1.,
               use_core_state: bool = True,
               prefetch_size: int = 2,
               replay_client: Optional[reverb.Client] = None,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None):
    """Initializes the Simfish R2D2 learner."""
    super().__init__(
        networks=networks,
        batch_size=batch_size,
        random_key=random_key,
        burn_in_length=burn_in_length,
        discount=discount,
        importance_sampling_exponent=importance_sampling_exponent,
        max_priority_weight=max_priority_weight,
        target_update_period=target_update_period,
        iterator=iterator,
        optimizer=optimizer,
        bootstrap_n=bootstrap_n,
        tx_pair=tx_pair,
        clip_rewards=clip_rewards,
        max_abs_reward=max_abs_reward,
        use_core_state=use_core_state,
        prefetch_size=prefetch_size,
        replay_client=replay_client,
        counter=counter,
        logger=logger)
    self.actions_mirror = actions_mirror

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
    key, subkey = jax.random.split(self._state.random_key[0])
    if jax.random.uniform(subkey) < 0.5:
      # Do a mirrored step first.
      mirrored = [True, False]
    else:
      # Do a regular step first.
      mirrored = [False, True]

    for mir in mirrored:
      start = time.time()
      if mir:
        self._state, priorities, metrics = self._sgd_step(
            self._state, reflect_samples(samples, self.actions_mirror))
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
