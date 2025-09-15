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

"""running R2D2 on fish environment."""

from absl import flags
from acme.agents.jax import r2d2
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import dm_env
import launchpad as lp
import json
from simulation.simfish_env import BaseEnvironment
from R2D2Network import make_r2d2_networks
from acme.utils import loggers
from typing import Optional
import dataclasses

from typing import Iterator, Optional, Sequence
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
import reverb
from acme.utils.loggers import tf_summary
from acme.utils.loggers import base as loggers_base
from acme.utils.loggers import filters
from acme.jax import networks as networks_lib

from hdf5_logger import HDF5Logger, EnvInfoKeep, SimpleEnvInfoKeep
from simfish_r2d2_learner import SimfishR2D2Learner
from acme.utils.loggers import aggregators
from acme.agents.jax.r2d2 import learning as r2d2_learning
from acme.agents.jax.r2d2 import networks as r2d2_networks
from define_actions import Actions
import optax


# Flags which modify the behavior of the launcher.
FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_integer(
    'num_actors', 15, 'Number of actors to use in the distributed setting. '
    'If run_distributed is False, this will be ignored.')
flags.DEFINE_string(
    'dir', '.', 'Results home directory.')
flags.DEFINE_string(
    'subdir', None, 'Subdirectory for this experiment.')
flags.DEFINE_string(
    'env_config_file', None,
    'Which environment config file to use.')
flags.DEFINE_integer(
    'seed', 42, 'Random seed to use for the experiment.')
flags.DEFINE_integer(
    'num_steps', 300_000_000, 'Number of steps to run the experiment for.')
flags.DEFINE_string(
  'actions_file', 'actions_all_bouts_with_null.h5',
  'File containing all possible actions.'
)

flags.mark_flag_as_required('env_config_file')
flags.mark_flag_as_required('subdir')

@dataclasses.dataclass
class SimfishR2D2Config(r2d2.R2D2Config):
  """Configuration options for R2D2 agent."""
  actions: Actions = Actions()

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
        actions_mirror=self._config.actions.get_opposing_dict(),
        logger=logger_fn('learner'))

def build_experiment_config(training_parameters: dict) -> experiments.ExperimentConfig:
  """Builds R2D2 experiment config which can be executed in different ways."""

  # Create an environment factory.
  def environment_factory(seed: int) -> dm_env.Environment:
    env_variables = json.load(open(training_parameters['env_config_file'], 'r'))
    return BaseEnvironment(env_variables=env_variables, seed=seed, actions=training_parameters['actions'].get_all_actions())

  # Configure the agent.
  config = SimfishR2D2Config(
      burn_in_length=training_parameters['burn_in_length'],
      trace_length=training_parameters['trace_length'],
      sequence_period=training_parameters['sequence_period'],
      min_replay_size=training_parameters['min_replay_size'],
      batch_size=training_parameters['batch_size'],
      prefetch_size=1,
      samples_per_insert=1.0,
      evaluation_epsilon=training_parameters['evaluation_epsilon'],
      learning_rate=training_parameters['learning_rate'],
      target_update_period=training_parameters['target_update_period'],
      variable_update_period=training_parameters['variable_update_period'],
      actions=training_parameters['actions']
  )
  network_factory = make_r2d2_networks
  builder = SimfishR2D2Builder(config)
  def make_hdf5_logger(label: str,
                           steps_key: Optional[str] = None,
                           task_instance: int = 0) -> loggers.Logger:
    del task_instance
    if steps_key is None:
      steps_key = f'{label}_steps'

    my_loggers = [HDF5Logger(label=label, wait_min=training_parameters['evaluator_waiting_minutes'], \
                             directory_or_file=training_parameters['directory'], \
                              add_uid=False)]
    
    # Dispatch to all writers and filter Nones and by time.
    logger = aggregators.Dispatcher(my_loggers, loggers_base.to_numpy)
    logger = filters.NoneFilter(logger)

    return logger

  def make_tfsummary_logger(label: str,
                            steps_key: Optional[str] = None,
                            task_instance: int = 0) -> loggers.Logger:
    if steps_key is None:
      steps_key = f'{label}_steps'

    my_loggers = [tf_summary.TFSummaryLogger(
        label=label, logdir=f"{training_parameters['directory']}/logs/training/{label}_{task_instance}",steps_key=steps_key)]
    
    # Dispatch to all writers and filter Nones and by time.
    logger = aggregators.Dispatcher(my_loggers, loggers_base.to_numpy)
    logger = filters.NoneFilter(logger)
    return logger
  
    
  def create_hdf5_logger_factory() -> loggers.LoggerFactory:
    return make_hdf5_logger
  
  def create_tfsummary_logger_factory() -> loggers.LoggerFactory:
    return make_tfsummary_logger
    
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
      observers=[SimpleEnvInfoKeep()],
      environment_factory=environment_factory,
      logger_factory=create_tfsummary_logger_factory(),
      evaluator_factories=eval_factories,
      seed=training_parameters['seed'],
      checkpointing=experiments.CheckpointingConfig(add_uid=False, directory=training_parameters['directory']),
      max_num_actor_steps=training_parameters['num_steps'],)
  

  return exp_config


def main(_):
  directory = f'{FLAGS.dir}/{FLAGS.subdir}'

  actions = Actions()
  actions.from_hdf5(FLAGS.actions_file)

  training_parameters = {'num_steps': FLAGS.num_steps,
                       'seed': FLAGS.seed,
                       'evaluator_waiting_minutes': 60,
                       'num_actors': FLAGS.num_actors,
                       'burn_in_length': 8,
                       'trace_length': 75,
                       'sequence_period': 20,
                       'min_replay_size': 10,
                       'batch_size': 32,
                       'evaluation_epsilon': 1e-3,
                       'learning_rate': 1e-4,
                       'target_update_period': 1200,
                       'variable_update_period': 100,
                       'directory': directory,
                       'env_config_file': FLAGS.env_config_file,
                       'actions': actions
                       }

  print(f"Running R2D2 with the following parameters: {training_parameters}")

  config = build_experiment_config(training_parameters)


  if FLAGS.run_distributed:

    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=training_parameters['num_actors'])
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
  else:
    print('Running single-threaded.')
    experiments.run_experiment(experiment=config, eval_every=1)


if __name__ == '__main__':
  app.run(main)
