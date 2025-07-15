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


# Flags which modify the behavior of the launcher.
flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_integer('seed', 1, 'Random seed (experiment).')
#flags.DEFINE_string('logging_dir', '~/acme', 'Directory to log to.')

FLAGS = flags.FLAGS



def build_experiment_config():
  """Builds R2D2 experiment config which can be executed in different ways."""
  batch_size = 32

  # Create an environment factory.
  def environment_factory(seed: int) -> dm_env.Environment:
    del seed
    env_variables = json.load(open('Environment/2_env.json', 'r'))
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
