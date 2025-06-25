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
from acme.utils.loggers import base


class HDF5Logger(base.Logger):
  """Standard HDF5 logger.


  """

  _open = open

  def __init__(
      self,
      directory_or_file: Union[str, TextIO] = '~/acme',
      label: str = '',
      add_uid: bool = True,
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

  def write(self, data: base.LoggingData):
  
    data = base.to_numpy(data)
    self.called += 1

    # write the fields of data into an hdf5 file
    with h5py.File(self.base_path + str(self.called) + '.hdf5', 'w') as f:
      for key, value in data.items():
        print(key)
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
    time.sleep(10 * 60)

  def close(self):
    pass
  

#   @property
#   def file_path(self) -> str:
#     return self._file.name
