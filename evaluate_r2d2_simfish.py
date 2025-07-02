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
from eval_agent import eval_agent
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
#flags.DEFINE_string('logging_dir', '~/acme', 'Directory to log to.')

FLAGS = flags.FLAGS



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
  builder = r2d2.R2D2Builder(config)
  
  exp_config = experiments.ExperimentConfig(
      builder=builder,
      network_factory=network_factory,
      #observers=[EnvInfoKeep()],
      environment_factory=environment_factory,
      #logger_factory=logger_factory,
      #evaluator_factories=eval_factories,
      seed=FLAGS.seed,
      checkpointing=experiments.CheckpointingConfig(add_uid=False),
      max_num_actor_steps=FLAGS.num_steps)
  

  return exp_config


def main(_):
  config = build_experiment_config()


  print('Running single-threaded.')
  eval_agent(experiment=config)


if __name__ == '__main__':
  app.run(main)
