# Copyright 2025 Asaph Zylbertal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
from typing import Optional, Sequence, List
import operator

import acme
from acme import core
from acme import specs

from acme.jax.experiments import config
from acme.tf import savers
from acme.utils import counting
import jax
import reverb
import dm_env
from hdf5_logger import HDF5Logger, EnvInfoKeep
from acme.utils import loggers
from acme.utils import observers as observers_lib
import tree
import numpy as np

def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)

class StateEnvLoop(acme.EnvironmentLoop):
  """An environment loop that keeps track of the state."""
    
  def __init__(
      self,
      environment: dm_env.Environment,
      actor: core.Actor,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      should_update: bool = True,
      label: str = 'environment_loop',
      observers: Sequence[observers_lib.EnvLoopObserver] = (),
  ):
    super().__init__(environment, actor, counter=counter, logger=logger,
                         observers=observers)
    

  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    episode_start_time = time.time()
    select_action_durations: List[float] = []
    env_step_durations: List[float] = []
    episode_steps: int = 0

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        self._environment.reward_spec())
    env_reset_start = time.time()
    timestep = self._environment.reset()
    env_reset_duration = time.time() - env_reset_start
    # Make the first observation.
    self._actor.observe_first(timestep)
    for observer in self._observers:
      # Initialize the observer with the current state of the env after reset
      # and the initial timestep.
      observer.observe_first(self._environment, timestep)

    # Run an episode.
    while not timestep.last():
      # Book-keeping.
      episode_steps += 1

      # Generate an action from the agent's policy.
      select_action_start = time.time()
      action = self._actor.select_action(timestep.observation)
      select_action_durations.append(time.time() - select_action_start)

      # Step the environment with the agent's selected action.
      env_step_start = time.time()
      timestep = self._environment.step(action)
      env_step_durations.append(time.time() - env_step_start)

      # Have the agent and observers observe the timestep.
      self._actor.observe(action, next_timestep=timestep)
      for observer in self._observers:
        # One environment step was completed. Observe the current state of the
        # environment, the current timestep and the action.
        observer.observe(self._environment, timestep, action,
                         actor_state=self._actor._state.recurrent_state.hidden)

      # Give the actor the opportunity to update itself.
      if self._should_update:
        self._actor.update()

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_return = tree.map_structure(operator.iadd,
                                          episode_return,
                                          timestep.reward)

    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - episode_start_time)
    result = {
        'episode_length': episode_steps,
        'episode_return': episode_return,
        'steps_per_second': steps_per_second,
        'env_reset_duration_sec': env_reset_duration,
        'select_action_duration_sec': np.mean(select_action_durations),
        'env_step_duration_sec': np.mean(env_step_durations),
    }
    result.update(counts)
    for observer in self._observers:
      result.update(observer.get_metrics())
    return result

def eval_agent(experiment: config.ExperimentConfig, directory: str, num_episodes: int = 100):

  key = jax.random.PRNGKey(experiment.seed)

  # Create the environment and get its spec.
  environment = experiment.environment_factory(experiment.seed)
  environment_spec = experiment.environment_spec or specs.make_environment_spec(
      environment)

  # Create the networks and policy.
  networks = experiment.network_factory(environment_spec)
  policy = config.make_policy(
      experiment=experiment,
      networks=networks,
      environment_spec=environment_spec,
      evaluation=False)

#   # Create the replay server and grab its address.
  replay_tables = experiment.builder.make_replay_tables(environment_spec,
                                                        policy)


  replay_server = reverb.Server(replay_tables, port=None)
  replay_client = reverb.Client(f'localhost:{replay_server.port}')

  parent_counter = counting.Counter(time_delta=0.)

  dataset = experiment.builder.make_dataset_iterator(replay_client)

  learner_key, key = jax.random.split(key)
  learner = experiment.builder.make_learner(
      random_key=learner_key,
      networks=networks,
      dataset=dataset,
      logger_fn=experiment.logger_factory,
      environment_spec=environment_spec,
      replay_client=replay_client,
      counter=counting.Counter(parent_counter, prefix='learner', time_delta=0.))


  # Create the evaluation actor and loop.
  eval_counter = counting.Counter(
      parent_counter, prefix='evaluator', time_delta=0.)
#   eval_logger = experiment.logger_factory('evaluator',
#                                           eval_counter.get_steps_key(), 0)
  eval_logger = HDF5Logger(label='model_evaluation', wait_min=0, directory_or_file=directory, add_uid=False)
  eval_policy = config.make_policy(
      experiment=experiment,
      networks=networks,
      environment_spec=environment_spec,
      evaluation=True)
  eval_actor = experiment.builder.make_actor(
      random_key=jax.random.PRNGKey(experiment.seed),
      policy=eval_policy,
      environment_spec=environment_spec,
      variable_source=learner)
  eval_loop = StateEnvLoop(
      environment,
      eval_actor,
      counter=eval_counter,
      logger=eval_logger,
      observers=[EnvInfoKeep()])

  eval_loop.run(num_episodes=num_episodes)

  environment.close()


