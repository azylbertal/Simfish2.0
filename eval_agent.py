# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applTrueicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runners used for executing local agents."""

import sys
import time
from typing import Optional, Sequence, Tuple

import acme
from acme import core
from acme import specs
from acme import types
from acme.jax import utils
from acme.jax.experiments import config
from acme.tf import savers
from acme.utils import counting
import jax
import reverb

from hdf5_logger import HDF5Logger, EnvInfoKeep


def eval_agent(experiment: config.ExperimentConfig):

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

  checkpointer = None
  if experiment.checkpointing is not None:
    checkpointing = experiment.checkpointing
    checkpointer = savers.Checkpointer(
        objects_to_save={'learner': learner},
        subdirectory='learner',
        time_delta_minutes=checkpointing.time_delta_minutes,
        directory=checkpointing.directory,
        add_uid=checkpointing.add_uid,
        max_to_keep=checkpointing.max_to_keep,
        keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
        checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
    )


  # Create the evaluation actor and loop.
  eval_counter = counting.Counter(
      parent_counter, prefix='evaluator', time_delta=0.)
#   eval_logger = experiment.logger_factory('evaluator',
#                                           eval_counter.get_steps_key(), 0)
  eval_logger = HDF5Logger(label='bbbbb', wait_min=0)
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
  eval_loop = acme.EnvironmentLoop(
      environment,
      eval_actor,
      counter=eval_counter,
      logger=eval_logger,
      observers=[EnvInfoKeep()])

  eval_loop.run(num_episodes=100)

  environment.close()


