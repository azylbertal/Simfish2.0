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

"""Example running R2D2 on fish environment."""

from absl import flags
from acme.agents.jax import r2d2
from absl import app
from acme.jax import experiments
from eval_agent import eval_agent
import dm_env
import json
from simulation.simfish_env import BaseEnvironment
from R2D2Network import make_r2d2_networks
from simulation.define_actions import Actions


# Flags which modify the behavior of the launcher.

flags.DEFINE_string(
    'dir', '.', 'Results home directory.')
flags.DEFINE_string(
    'subdir', None, 'Subdirectory for this experiment.')
flags.DEFINE_string(
    'log_subdir', None, 'Subdirectory for results.')

flags.DEFINE_string(
    'env_config_file', None,
    'Which environment config file to use.')
flags.DEFINE_integer(
    'seed', 42, 'Random seed to use for the experiment.')
flags.DEFINE_integer(
    'num_episodes', 100, 'Number of episodes to run.')
flags.DEFINE_string(
  'actions_file', 'actions_all_bouts_with_null.h5',
  'File containing all possible actions.'
)

flags.mark_flag_as_required('env_config_file')
flags.mark_flag_as_required('subdir')
flags.mark_flag_as_required('log_subdir')

FLAGS = flags.FLAGS

actions = Actions()
actions.from_hdf5('actions_all_bouts_with_null.h5')
actions_mirror = actions.get_opposing_dict()


def build_experiment_config(directory, env_config_file):
  """Builds R2D2 experiment config which can be executed in different ways."""
  batch_size = 32

  # Create an environment factory.
  def environment_factory(seed: int) -> dm_env.Environment:
    env_variables = json.load(open(env_config_file, 'r'))
    return BaseEnvironment(env_variables=env_variables, seed=seed, actions=actions.get_all_actions())

  # Configure the agent.
  config = r2d2.R2D2Config(
      burn_in_length=8,
      trace_length=75,
      sequence_period=20,
      min_replay_size=10,#_000,
      batch_size=batch_size,
      prefetch_size=1,
      samples_per_insert=1.0,
      evaluation_epsilon=0,#1e-3,
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
      checkpointing=experiments.CheckpointingConfig(add_uid=False, directory=directory),
      max_num_actor_steps=0)
  

  return exp_config


def main(_):
  directory = f'{FLAGS.dir}/{FLAGS.subdir}'

  config = build_experiment_config(directory, FLAGS.env_config_file)


  print('Running single-threaded.')
  eval_agent(experiment=config, directory=directory, num_episodes=FLAGS.num_episodes, log_subdir=FLAGS.log_subdir)


if __name__ == '__main__':
  app.run(main)
