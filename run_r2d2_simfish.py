# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
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

"""Example running R2D2 on fish environment."""

from absl import flags
from acme.agents.jax import r2d2
# import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import dm_env
import launchpad as lp
import json
from Environment.SimFishEnv import BaseEnvironment
from Environment.DummyEnv import DummyEnv
from R2D2Network import make_r2d2_networks
from acme.utils import loggers
from typing import Optional
from acme.utils.observers.env_info import EnvInfoObserver
import numpy as np
from typing import Dict
from acme.utils.observers import base

import logging
from typing import Any, Callable, Dict, Generic, Iterator, Optional, Sequence
from acme import core
from acme import environment_loop
from acme import specs
from acme.agents.jax import builders
from acme.jax import types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
import jax
from typing_extensions import Protocol
import reverb

from acme.utils.loggers import base as loggers_base
from acme.utils.loggers import filters
from acme.jax import networks as networks_lib

from hdf5_logger import HDF5Logger
from simfish_r2d2_learner import SimfishR2D2Learner
from acme.utils.loggers import aggregators
from acme.agents.jax.r2d2 import learning as r2d2_learning
from acme.agents.jax.r2d2 import networks as r2d2_networks
import optax

# import matplotlib
# matplotlib.use('QtAgg')
# Flags which modify the behavior of the launcher.
flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('env_name', 'Pong', 'What environment to run.')
flags.DEFINE_integer('seed', 1, 'Random seed (experiment).')
flags.DEFINE_integer('num_steps', 25_000_000,
                     'Number of environment steps to run for.')
flags.DEFINE_string('logging_dir', '~/acme', 'Directory to log to.')

FLAGS = flags.FLAGS
class SimfishR2D2Builder(r2d2.R2D2Builder):
  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: r2d2_networks.R2D2Networks,
      dataset: Iterator[r2d2_learning.R2D2ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec

    # The learner updates the parameters (and initializes them).
    return SimfishR2D2Learner(
        networks=networks,
        batch_size=self._batch_size_per_device,
        random_key=random_key,
        burn_in_length=self._config.burn_in_length,
        discount=self._config.discount,
        importance_sampling_exponent=(
            self._config.importance_sampling_exponent),
        max_priority_weight=self._config.max_priority_weight,
        target_update_period=self._config.target_update_period,
        iterator=dataset,
        optimizer=optax.adam(self._config.learning_rate),
        bootstrap_n=self._config.bootstrap_n,
        tx_pair=self._config.tx_pair,
        clip_rewards=self._config.clip_rewards,
        replay_client=replay_client,
        counter=counter,
        logger=logger_fn('learner'))

class EnvInfoKeep(base.EnvLoopObserver):
  """An observer that collects and accumulates scalars from env's info."""

  def __init__(self):
    self._metrics = None

  def _accumulate_metrics(self, env: dm_env.Environment, obs) -> None:
    if not hasattr(env, 'get_info'):
      return
    info = getattr(env, 'get_info')()
    info['action'] = [int(obs.action)]
    info['vis_observation'] = [obs.observation[0]]
    info['internal_state'] = [obs.observation[1]]
    if not info:
      return
    for k, v in info.items():
      self._metrics[k] = self._metrics.get(k, []) + v

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state."""
    sediment = env.board.global_sediment_grating
    self._metrics = {'sediment': [sediment]}
    self._accumulate_metrics(env, timestep.observation)

  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    self._accumulate_metrics(env, timestep.observation)

  def get_metrics(self) -> Dict[str, base.Number]:
    """Returns metrics collected for the current episode."""
    return self._metrics

def build_experiment_config():
  """Builds R2D2 experiment config which can be executed in different ways."""
  batch_size = 32

  # The env_name must be dereferenced outside the environment factory as FLAGS
  # cannot be pickled and pickling is necessary when launching distributed
  # experiments via Launchpad.
  env_name = FLAGS.env_name

  # Create an environment factory.
  def environment_factory(seed: int) -> dm_env.Environment:
    del seed
    # return DummyEnv()
    env_variables = json.load(open('Environment/1_env.json', 'r'))
    return BaseEnvironment(env_variables=env_variables)

  # Configure the agent.
  config = r2d2.R2D2Config(
      burn_in_length=8,
      trace_length=40,
      sequence_period=20,
      min_replay_size=10,#_000,
      batch_size=batch_size,
      prefetch_size=1,
      samples_per_insert=1.0,
      evaluation_epsilon=1e-3,
      learning_rate=1e-4,
      target_update_period=1200,
      variable_update_period=100,
  )
  network_factory = make_r2d2_networks
  builder = SimfishR2D2Builder(config)
  def make_hdf5_logger(label: str,
                           steps_key: Optional[str] = None,
                           task_instance: int = 0) -> loggers.Logger:
    del task_instance
    if steps_key is None:
      steps_key = f'{label}_steps'





#def make_default_logger(
#     label: str,
#     save_data: bool = True,
#     time_delta: float = 1.0,
#     asynchronous: bool = False,
#     print_fn: Optional[Callable[[str], None]] = None,
#     serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = base.to_numpy,
#     steps_key: str = 'steps',
# ) -> base.Logger:
  # """Makes a default Acme logger.

  # Args:
  #   label: Name to give to the logger.
  #   save_data: Whether to persist data.
  #   time_delta: Time (in seconds) between logging events.
  #   asynchronous: Whether the write function should block or not.
  #   print_fn: How to print to terminal (defaults to print).
  #   serialize_fn: An optional function to apply to the write inputs before
  #     passing them to the various loggers.
  #   steps_key: Ignored.

  # Returns:
  #   A logger object that responds to logger.write(some_dict).
  # """
  #del steps_key
    loggers = [HDF5Logger(label=label)]
    
    # Dispatch to all writers and filter Nones and by time.
    logger = aggregators.Dispatcher(loggers, loggers_base.to_numpy)
    logger = filters.NoneFilter(logger)

    return logger





  def create_hdf5_logger_factory() -> loggers.LoggerFactory:
    return make_hdf5_logger
  
  
  def eval_policy_factory(networks: builders.Networks,
                          environment_spec: specs.EnvironmentSpec,
                          evaluation: bool) -> builders.Policy:
    del evaluation
    # The config factory has precedence until all agents are migrated to use
    # builder.make_policy
    return builder.make_policy(
        networks=networks,
        environment_spec=environment_spec,
        evaluation=True)

  
  
  logger_factory = create_hdf5_logger_factory()
  
  
  def evaluator_factory(
    environment_factory: types.EnvironmentFactory,
    network_factory: experiments.NetworkFactory[builders.Networks],
    policy_factory: experiments.PolicyFactory[builders.Networks, builders.Policy],
    logger_factory: loggers.LoggerFactory,
    observers: Sequence[observers_lib.EnvLoopObserver] = (),
) -> experiments.EvaluatorFactory[builders.Policy]:
    """Returns a default evaluator process."""

    def evaluator(
        random_key: types.PRNGKey,
        variable_source: core.VariableSource,
        counter: counting.Counter,
        make_actor: experiments.MakeActorFn[builders.Policy],
    ):
      """The evaluation process."""
      # Create environment and evaluator networks
      environment_key, actor_key = jax.random.split(random_key)
      # Environments normally require uint32 as a seed.
      environment = environment_factory(utils.sample_uint32(environment_key))
      environment_spec = specs.make_environment_spec(environment)
      networks = network_factory(environment_spec)
      policy = policy_factory(networks, environment_spec, True)
      actor = make_actor(actor_key, policy, environment_spec, variable_source)

      # Create logger and counter.
      counter = counting.Counter(counter, 'evaluator')
      logger = logger_factory('evaluator', 'actor_steps', 0)

      # Create the run loop and return it.
      return environment_loop.EnvironmentLoop(
          environment, actor, counter, logger, observers=observers)

    return evaluator

  
  
  eval_factories = [
        evaluator_factory(
            environment_factory=environment_factory,
            network_factory=network_factory,
            policy_factory=eval_policy_factory,
            logger_factory=logger_factory,
            observers=[EnvInfoKeep()])
    ]
  
  exp_config = experiments.ExperimentConfig(
      builder=builder,
      network_factory=network_factory,
      #observers=[EnvInfoKeep()],
      environment_factory=environment_factory,
      #logger_factory=logger_factory,
      evaluator_factories=eval_factories,
      seed=FLAGS.seed,
      checkpointing=experiments.CheckpointingConfig(add_uid=True),
      max_num_actor_steps=FLAGS.num_steps)
  

  return exp_config


def main(_):
  config = build_experiment_config()


  if FLAGS.run_distributed:

    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=15 if lp_utils.is_local_run() else 3)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
  else:
    print('Running single-threaded.')
    experiments.run_experiment(experiment=config, eval_every=1)


if __name__ == '__main__':
  app.run(main)
