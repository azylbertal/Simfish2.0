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
            raise ValueError('Expected input BPE or PE. Got rank %d' % inputs_rank) #BPE is batch, position, eye

        outputs = self._network(inputs)

        if batched_inputs:
            return jnp.reshape(outputs, [outputs.shape[0], -1])
        else:
            return jnp.reshape(outputs, [-1])
        
class DeepSimfishTorso(hk.Module):
  """Deep torso for Atari, from the IMPALA paper."""

  def __init__(
      self,
      hidden_sizes: Sequence[int] = (256,),
      name: str = 'deep_simfish_torso'):
    super().__init__(name=name)
    self.retina = retina()
    # Make sure to activate the last layer as this torso is expected to feed
    # into the rest of a bigger network.
    self.mlp_head = hk.nets.MLP(output_sizes=hidden_sizes, activate_final=True)

  def __call__(self, x: [jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    left_eye = x[0][:, :, :, 0]
    right_eye = x[0][:, ::-1, :, 1]
    left_eye = self.retina(left_eye)
    right_eye = self.retina(right_eye)
    output = jnp.concatenate([left_eye, right_eye], axis=-1)
    #output = x
    output = jax.nn.relu(output)
    #output = hk.Flatten(preserve_dims=-3)(output)
    output = self.mlp_head(output)
    # concatenate internal state
    if len(x) > 1:

        output = jnp.concatenate([output, x[1]], axis=-1)
    return output


class R2D2SimfishNetwork(hk.RNNCore):
  """Based on aa duelling recurrent network for use with Atari observations as seen in R2D2.

  See https://openreview.net/forum?id=r1lyTjAqYX for more information.
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
