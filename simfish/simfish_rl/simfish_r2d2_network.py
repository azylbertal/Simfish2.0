
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

"""R2D2 Network Architecture for Simfish RL Environment.
This module implements a Recurrent Experience Replay in Distributed Reinforcement Learning (R2D2)
network architecture tailored for the Simfish environment. The network processes visual observations
from two eyes along with action and reward information.
Architecture Overview:
----------------------

1. **Input Processing (OAREmbedding)**:
  - Takes observation-action-reward (OAR) tuples
  - Processes observations through DeepSimfishTorso
  - One-hot encodes actions
  - Normalizes rewards using tanh to [-1, 1]
  - Concatenates all features
2. **Visual Processing (DeepSimfishTorso)**:
  - Splits binocular input into left and right eye channels
  - Each eye processes through a retina module:
    * Conv1D(8 filters, kernel=3, stride=1) + ReLU
    * Conv1D(8 filters, kernel=5, stride=2) + ReLU
    * Conv1D(8 filters, kernel=5, stride=2) + ReLU
  - Concatenates left and right eye features
  - Passes through MLP head with configurable hidden sizes (default: 32)
  - Optionally concatenates internal state information
3. **Recurrent Core (R2D2SimfishNetwork)**:
  - LSTM with 64 hidden units for temporal processing
  - Maintains hidden and cell states across timesteps
4. **Action Value Estimation**:
  - Duelling MLP head with 64 hidden units
  - Separates state value and action advantage streams
  - Outputs Q-values for all actions
The network supports both single-step and batch unroll operations, making it suitable
for distributed training with experience replay.
Classes:
--------
- OAREmbedding: Embeds observation, action, and reward inputs
- Flatten: Utility module for flattening tensors
- retina: 1D convolutional network for processing eye input
- DeepSimfishTorso: Binocular visual processing network
- R2D2SimfishNetwork: Main recurrent network with LSTM core and duelling head
Functions:
----------
- make_r2d2_networks: Factory function to create R2D2 networks from environment specs
"""


from acme import specs
from acme.jax import networks as networks_lib
from acme.jax.networks import duelling

import haiku as hk
import jax.numpy as jnp
import jax
from typing import Optional, Tuple, Sequence
from acme.wrappers import observation_action_reward
from acme.jax.networks import base
import dataclasses

@dataclasses.dataclass
class OAREmbedding(hk.Module):
  """Module for embedding (observation, action, reward) inputs together."""

  torso: hk.SupportsCall
  num_actions: int

  def __call__(self, inputs: observation_action_reward.OAR) -> jnp.ndarray:
    """Embed each of the (observation, action, reward) inputs & concatenate."""

    # Add dummy batch dimension to observation if necessary.
    # This is needed because Conv2D assumes a leading batch dimension, i.e.
    # that inputs are in [B, H, W, C] format.
    expand_obs = len(inputs.observation[0].shape) == 3
    if expand_obs:
        
        inputs = inputs._replace(
          observation=[jnp.expand_dims(inputs.observation[0], axis=0), jnp.expand_dims(inputs.observation[1], axis=0)])
    features = self.torso(inputs.observation)  # [T?, B, D]
    if expand_obs:
      features = jnp.squeeze(features, axis=0)

    # Do a one-hot embedding of the actions.
    action = jax.nn.one_hot(
        inputs.action, num_classes=self.num_actions)  # [T?, B, A]

    # Map rewards -> [-1, 1].
    reward = jnp.tanh(inputs.reward)

    # Add dummy trailing dimensions to rewards if necessary.
    while reward.ndim < action.ndim:
      reward = jnp.expand_dims(reward, axis=-1)

    # Concatenate on final dimension.
    embedding = jnp.concatenate(
        [features, action, reward], axis=-1)  # [T?, B, D+A+1]

    return embedding

class Flatten(hk.Module):
    def __init__(self, name="Flatten"):
        super().__init__(name=name)

    def __call__(self, x):
        # the [None, ...] is important because acme expects (1, feature_dim) shape
        return x.flatten()[None, ...]

class retina(hk.Module):
    """
  A convolutional neural network module for processing retinal input data.
  This module implements a 1D convolutional network designed to process retinal 
  information, consisting of three convolutional layers with ReLU activations.
  The network architecture:
  - Conv1D: 8 filters, kernel size 3, stride 1
  - ReLU activation
  - Conv1D: 8 filters, kernel size 5, stride 2
  - ReLU activation
  - Conv1D: 8 filters, kernel size 5, stride 2
  - ReLU activation
  Args:
    inputs (jnp.ndarray): Input array with shape either:
      - (position, channel) - PE format for single sample
      - (batch, position, channel) - BPE format for batched samples
  Returns:
    jnp.ndarray: Flattened output features with shape:
      - (batch, features) for batched inputs
      - (features,) for single sample inputs
  Raises:
    ValueError: If input rank is not 2 (PC) or 3 (BPC).
  Example:
    >>> retina_net = retina()
    >>> # Single sample: (position, channel)
    >>> output = retina_net(jnp.ones((100, 2)))
    >>> # Batched: (batch, position, channel)
    >>> output = retina_net(jnp.ones((32, 100, 2)))
  """
   
    def __init__(
         self,

    ):
       super().__init__()
       self._network = hk.Sequential([
           hk.Conv1D(8, 3, 1), jax.nn.relu,
           hk.Conv1D(8, 5, 2), jax.nn.relu,
           hk.Conv1D(8, 5, 2), jax.nn.relu
       ])
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        inputs_rank = jnp.ndim(inputs)
        batched_inputs = inputs_rank == 3
        if inputs_rank < 2 or inputs_rank > 3:
            raise ValueError('Expected input BPC or PC. Got rank %d' % inputs_rank) #BPC is batch, position, channel

        outputs = self._network(inputs)

        if batched_inputs:
            return jnp.reshape(outputs, [outputs.shape[0], -1])
        else:
            return jnp.reshape(outputs, [-1])
        
class DeepSimfishTorso(hk.Module):
  """Neural network torso for processing binocular visual input in Simfish.

  This class processes left and right eye inputs through a shared retina network,
  concatenates the processed outputs, and passes them through an MLP head. Optionally
  concatenates internal state if provided.

  Args:
    hidden_sizes: Sequence of integers specifying the sizes of hidden layers in the
      MLP head. Defaults to (256,).
    name: Name of the module. Defaults to 'deep_simfish_torso'.

  Input:
    x: A list of jnp.ndarray where:
      - x[0]: Visual input with shape (batch, position, channel, eye) where eye
        dimension has size 2 (left=0, right=1). Right eye positions are reversed
        to match left eye orientation.
      - x[1] (optional): Internal state to concatenate with processed visual output.

  Returns:
    jnp.ndarray: Processed output combining binocular visual features from the MLP
      head, optionally concatenated with internal state if provided.

  Note:
    The right eye input is reversed along the position axis to align with the left
    eye's spatial orientation before processing through the shared retina network.
  """

  def __init__(
      self,
      hidden_sizes: Sequence[int] = (256,),
      name: str = 'deep_simfish_torso'):
    super().__init__(name=name)
    self.retina = retina()
    # Make sure to activate the last layer as this torso is expected to feed
    # into the rest of a bigger network.
    self.mlp_head = hk.nets.MLP(output_sizes=hidden_sizes, activate_final=True)

  def __call__(self, x: list[jnp.ndarray]) -> jnp.ndarray:
    left_eye = x[0][:, :, :, 0] # x[0] is batch, position, channel, eye
    right_eye = x[0][:, ::-1, :, 1] # Reverse the right eye position order to match left eye
    left_eye = self.retina(left_eye)
    right_eye = self.retina(right_eye)
    output = jnp.concatenate([left_eye, right_eye], axis=-1)
    output = jax.nn.relu(output)
    output = self.mlp_head(output)
    # concatenate with internal state
    if len(x) > 1:

        output = jnp.concatenate([output, x[1]], axis=-1)
    return output


class R2D2SimfishNetwork(hk.RNNCore):
  """R2D2 network architecture for Simfish environment.

  This network implements the R2D2 (Recurrent Replay Distributed DQN) architecture
  tailored for the Simfish environment. It combines observation, action, and reward
  embeddings with an LSTM core and a duelling DQN head.

  The network processes sequences of (observation, action, reward) tuples through:
  1. An embedding layer (OAREmbedding with DeepSimfishTorso)
  2. An LSTM core for temporal dependencies
  3. A duelling MLP head for Q-value estimation

  Attributes:
    _embed: OAREmbedding layer that processes observations, actions, and rewards
      using a DeepSimfishTorso with hidden sizes [32].
    _core: LSTM layer with 64 hidden units for recurrent processing.
    _duelling_head: Duelling DQN head with hidden sizes [64] for Q-value estimation.
    _num_actions: Number of possible actions in the environment.

  Methods:
    __call__: Processes a single timestep of inputs through the network.
    initial_state: Returns the initial LSTM hidden state for a given batch size.
    unroll: Efficiently processes a sequence of inputs using static unrolling.
  """


  def __init__(self, num_actions: int):
    super().__init__(name='r2d2_simfish_network')
    self._embed = OAREmbedding(
        DeepSimfishTorso(hidden_sizes=[32]), num_actions)
    self._core = hk.LSTM(64)
    self._duelling_head = duelling.DuellingMLP(num_actions, hidden_sizes=[64])
    self._num_actions = num_actions

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    embeddings = self._embed(inputs)  # [B, D+A+1]
    core_outputs, new_state = self._core(embeddings, state)
    q_values = self._duelling_head(core_outputs)
    return q_values, new_state

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
    return self._core.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    embeddings = hk.BatchApply(self._embed)(inputs)  # [T, B, D+A+1]
    core_outputs, new_states = hk.static_unroll(self._core, embeddings, state)
    q_values = hk.BatchApply(self._duelling_head)(core_outputs)  # [T, B, A]
    return q_values, new_states






def make_r2d2_networks(env_spec: specs.EnvironmentSpec) -> R2D2SimfishNetwork:
    """Builds default R2D2 networks for simple networks."""

    def make_core_module() -> R2D2SimfishNetwork:
        return R2D2SimfishNetwork(env_spec.actions.num_values)

    return networks_lib.make_unrollable_network(env_spec, make_core_module)
