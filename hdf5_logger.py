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

"""A simple HDF5 logger.

Warning: Does not support preemption.
"""

import h5py
import numpy as np
import os
import time
from typing import TextIO, Union

from acme.utils import paths
from acme.utils.loggers import base as base_loggers
from acme.utils.observers import base as base_observers
import dm_env

from typing import Any, Callable, Dict, Generic, Iterator, Optional, Sequence

class EnvInfoKeep(base_observers.EnvLoopObserver):
  """An observer that collects and accumulates scalars from env's info."""

  def __init__(self):
    self._metrics = None

  def _accumulate_metrics(self, env: dm_env.Environment, obs, actor_state: Optional[np.ndarray] = None) -> None:
    if not hasattr(env, 'get_info'):
      return
    info = getattr(env, 'get_info')()
    info['action'] = [int(obs.action)]
    info['vis_observation'] = [obs.observation[0]]
    info['internal_state'] = [obs.observation[1]]
    if actor_state is not None:
      info['actor_state'] = [actor_state]
    if not info:
      return
    for k, v in info.items():
      self._metrics[k] = self._metrics.get(k, []) + v

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state."""
    sediment = env.board.global_sediment_grating
    self._metrics = {'sediment': [sediment], 'salt_location': [env.salt_location]}
    self._accumulate_metrics(env, timestep.observation)

  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray, actor_state: Optional[np.ndarray] = None) -> None:
    """Records one environment step."""
    self._accumulate_metrics(env, timestep.observation, actor_state)

  def get_metrics(self) -> Dict[str, base_observers.Number]:
    """Returns metrics collected for the current episode."""
    return self._metrics

class HDF5Logger(base_loggers.Logger):
  """Standard HDF5 logger.


  """

  _open = open

  def __init__(
      self,
      directory_or_file: Union[str, TextIO] = '~/acme',
      label: str = '',
      add_uid: bool = True,
      wait_min: float = 10,
  ):
    """Instantiates the logger.

    Args:
      directory_or_file: Either a directory path as a string, or a file TextIO
        object.
      label: Extra label to add to logger. This is added as a suffix to the
        directory.
      add_uid: Whether to add a UID to the file path. See `paths.process_path`
        for details.
    """

    self._add_uid = add_uid

    directory = paths.process_path(
          directory_or_file, 'logs', label, add_uid=self._add_uid)
    self.base_path = os.path.join(directory, 'logs_')
    self.called = 0
    self.wait_min = wait_min

  def write(self, data: base_loggers.LoggingData):
  
    data = base_loggers.to_numpy(data)
    self.called += 1

    # write the fields of data into an hdf5 file
    with h5py.File(self.base_path + str(self.called) + '.hdf5', 'w') as f:
      for key, value in data.items():
        # if key contains 'prey'...
        if 'prey' in key:
          # value is a list of lists with variable length. find the maximal length
            max_prey_num = max([len(x) for x in value])
            # create a 2d numpy array with the maximal length
            prey_array = np.zeros((len(value), max_prey_num))
            # fill with nans
            prey_array.fill(np.nan)
            # fill the array with the values from the list of lists
            for i in range(len(value)):
              prey_array[i, :len(value[i])] = value[i]
            # save the array
            f.create_dataset(key, data=prey_array)
        else:
            f.create_dataset(key, data=value)
    # wait to space out the writes
    time.sleep(self.wait_min * 60)

  def close(self):
    pass
  

#   @property
#   def file_path(self) -> str:
#     return self._file.name
