# Copyright 2025 Asaph Zylbertal
"""
Evaluation script for R2D2 agent on SimFish environment.
This module provides functionality to evaluate a trained R2D2 reinforcement learning
agent on the SimFish simulation environment. It loads a trained model from a checkpoint
and runs evaluation episodes to assess agent performance.
Parameters:
    --dir (str): Results home directory containing the trained models as subdirectories.
    --subdir (str): Subdirectory for this experiment, typically containing one agent.
    --log_subdir (str): Subdirectory where evaluation results will be saved under dir/subdir/logs. Required.
    --env_config_file (str): Path to the YAML configuration file that defines
        environment parameters. Required.
    --seed (int): Random seed to use for reproducibility of the evaluation.
        Default: 42
    --num_episodes (int): Number of evaluation episodes to run.
        Default: 100
Example:
    python evaluate_r2d2_simfish.py \\
        --dir=./results \\
        --subdir=agent_001 \\
        --log_subdir=eval_logs \\
        --env_config_file=configs/env_config.yaml \\
        --seed=42 \\
        --num_episodes=100
Functions:
    read_config_file: Reads and flattens a YAML configuration file.
    build_experiment_config: Constructs the R2D2 experiment configuration for evaluation.
    main: Entry point that orchestrates the evaluation process.
"""
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


from absl import flags
from acme.agents.jax import r2d2
from absl import app
from acme.jax import experiments
from simfish_rl import eval_agent
import dm_env
import yaml
from simulation import BaseEnvironment, Actions
from simfish_rl import make_r2d2_networks
from utils import read_config_file

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


flags.mark_flag_as_required('env_config_file')
flags.mark_flag_as_required('subdir')
flags.mark_flag_as_required('log_subdir')

FLAGS = flags.FLAGS

actions = Actions()
actions.from_hdf5('simulation/actions_all_bouts_with_null.h5')
actions_mirror = actions.get_opposing_dict()

def build_experiment_config(directory, env_config_file):
  """Builds R2D2 experiment config which can be executed in different ways."""
  batch_size = 32

  # Create an environment factory.
  def environment_factory(seed: int) -> dm_env.Environment:
    env_variables = read_config_file(env_config_file)
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
