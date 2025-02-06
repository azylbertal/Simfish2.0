# Set up dev env

This needs Linux env, since `jax` doesn't work on Windows.
It's recommended to do these on RIS, since `cupy` will need to use GPU anyway.

1. Set up a conda env with `conda env create -f environment.yml`
2. `git clone https://github.com/google-deepmind/acme`, check that the jax version requirement is consistent.
3. activate the env and then `cd` to the `acme` directory
4. `pip install .[jax,tf,testing,envs]`
5. `conda install -c "nvidia/label/cuda-11.6.2" cuda-nvcc`
6. If you see the error
```bash
"envs/acme/lib/python3.9/site-packages/haiku/_src/dot.py", line 29, in <module>
    from jax.extend import linear_util as lu
ModuleNotFoundError: No module named 'jax.extend'
```
you may need to do `pip install dm-haiku==0.0.10` to downgrade `haiku` version

## runtime

### python.so file
During each runtime, it's possible that you get the error `ImportError: libpython3.9.so.1.0: cannot open shared object file: No such file or directory`.
In this case, you need to **setup `LD_LIBRARY_PATH` manually** by doing `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/miniconda3/envs/acme/lib` or `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/robert/miniconda3/envs/acme/lib` depending on where exactly the location of your `acme`.

### cpu only
Note: for local debugging purpose, you may run it on CPU-only by doing `export CUDA_VISIBLE_DEVICES=-1`, and set the math library in `drawing_board.py` amd `eye.py` to `np`.

### 64-bit with JAX
do `export JAX_ENABLE_X64=True` to enable 64-bit integers and floats, see more [here](https://github.com/google/jax#current-gotchas).
Personally I think it's unnecessary to use 64-bit precision, but I think parts of `acme` (error is from `dm-reverb`) still expects `int64` instead of `int32`.

## Test GPU capability

### `jax`

```Python
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
```

[test matrix multiplication](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)

### `cupy`

```Python
import cupy as cp
x_gpu = cp.array([1, 2, 3])
x_gpu.device
```

### `tensorflow`

```Python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

[test matrix multiplication](https://www.tensorflow.org/api_docs/python/tf/math/multiply)

## Example from `ACME` --- `r2d2.py`

`examples/baselines/rl_discrete/run_r2d2.py`

It should work now.

# Notes for runtime debug

When running `python run_r2d2_simfish.py --norun_distributed`, encounter some errors which required decent amount of time to resolve...
Currently both errors are resolved.

## Error 1

```
defining drawing board
defining physics
/data/miniconda3/envs/acme/lib/python3.9/site-packages/acme/jax/utils.py:79: UserWarning: Explicitly requested
 dtype int64 requested in zeros is not available, and will be truncated to dtype int32. To enable more dtypes,
 set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://git
hub.com/google/jax#current-gotchas for more.
  return jax.tree_map(lambda x: jnp.zeros(x.shape, dtype or x.dtype), nest)
/data/miniconda3/envs/acme/lib/python3.9/site-packages/acme/jax/utils.py:79: UserWarning: Explicitly requested
 dtype float64 requested in zeros is not available, and will be truncated to dtype float32. To enable more dty
pes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https:/
/github.com/google/jax#current-gotchas for more.
  return jax.tree_map(lambda x: jnp.zeros(x.shape, dtype or x.dtype), nest)
[reverb/cc/platform/tfrecord_checkpointer.cc:162]  Initializing TFRecordCheckpointer in /tmp/119361.tmpdir/tmp
cvodii7c.
[reverb/cc/platform/tfrecord_checkpointer.cc:552] Loading latest checkpoint from /tmp/119361.tmpdir/tmpcvodii7
c
[reverb/cc/platform/default/server.cc:71] Started replay server on port 40853
[reverb/cc/client.cc:165] Sampler and server are owned by the same process (3702) so Table priority_table is a
ccessed directly without gRPC.
I1012 13:44:45.876254 139910434113344 csv.py:76] Logging to /home/robert.wong/acme/20231012-134242/logs/learne
r/logs.csv
Traceback (most recent call last):
  File "/scratch/RL_fish/simfish_env/run_r2d2_simfish.py", line 101, in <module>
    app.run(main)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/absl/app.py", line 308, in run
    _run_main(main, args)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/absl/app.py", line 254, in _run_main
    sys.exit(main(argv))
  File "/scratch/RL_fish/simfish_env/run_r2d2_simfish.py", line 97, in main
    experiments.run_experiment(experiment=config)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/acme/jax/experiments/run_experiment.py", line 9
3, in run_experiment
    learner = experiment.builder.make_learner(
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/acme/agents/jax/r2d2/builder.py", line 144, in 
make_learner
    return r2d2_learning.R2D2Learner(
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/acme/agents/jax/r2d2/learning.py", line 223, in
 __init__
    initial_params = networks.init(key_init)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/multi_transform.py", line 2[52/1854]
it_fn
    params, state = f.init(rng, *args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/transform.py", line 427, in init_fn
    f(*args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/multi_transform.py", line 145, in in
it_fn
    return f()[0](*args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/acme/jax/networks/base.py", line 144, in init
    return model(dummy_observation, model.initial_state(None))
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/module.py", line 465, in wrapped
    out = f(*args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/module.py", line 306, in run_interce
ptors
    return bound_method(*args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/acme/jax/networks/atari.py", line 166, in __cal
l__
    embeddings = self._embed(inputs)  # [B, D+A+1]                                                   [33/1854]
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/module.py", line 465, in wrapped
    out = f(*args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/module.py", line 306, in run_interce
ptors
    return bound_method(*args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/acme/jax/networks/embedding.py", line 42, in __
call__
    features = self.torso(inputs.observation)  # [T?, B, D]
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/module.py", line 465, in wrapped
    out = f(*args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/module.py", line 306, in run_interce
ptors
    return bound_method(*args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/acme/jax/networks/atari.py", line 102, in __cal
l__
    output = self.resnet(x)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/module.py", line 465, in wrapped
    out = f(*args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/module.py", line 306, in run_interce
ptors
    return bound_method(*args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/acme/jax/networks/resnet.py", line 149, in __ca
ll__
    output = make_downsampling_layer(strategy, num_channels)(output)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/module.py", line 465, in wrapped
    out = f(*args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/module.py", line 306, in run_interce
ptors
    return bound_method(*args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/basic.py", line 125, in __call__
    out = layer(out, *args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/module.py", line 465, in wrapped
    out = f(*args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/module.py", line 306, in run_interce
ptors
    return bound_method(*args, **kwargs)
  File "/data/miniconda3/envs/acme/lib/python3.9/site-packages/haiku/_src/conv.py", line 179, in __call__
    raise ValueError(f"Input to ConvND needs to have rank in {allowed_ranks},"
ValueError: Input to ConvND needs to have rank in [3, 4], but input has shape (278, 3).
[reverb/cc/platform/default/server.cc:84] Shutting down replay server
```

## Error 2

```Python
Traceback (most recent call last):
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/reverb/structured_writer.py", line 94, in append
    self._writer.AppendPartial(flat_data)
ValueError: Tensor of wrong dtype provided for column 1. Got int32 but expected int64.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/c/Users/rober/OneDrive - Washington University in St. Louis/Documents/lab/RL/SimFish/simfish_env/run_r2d2_simfish.py", line 101, in <module>
    app.run(main)
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/absl/app.py", line 308, in run
    _run_main(main, args)
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/absl/app.py", line 254, in _run_main
    sys.exit(main(argv))
  File "/mnt/c/Users/rober/OneDrive - Washington University in St. Louis/Documents/lab/RL/SimFish/simfish_env/run_r2d2_simfish.py", line 97, in main
    experiments.run_experiment(experiment=config)
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/acme/jax/experiments/run_experiment.py", line 177, in run_experiment
    steps += train_loop.run(num_steps=num_steps)
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/acme/environment_loop.py", line 198, in run
    result = self.run_episode()
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/acme/environment_loop.py", line 122, in run_episode
    self._actor.observe(action, next_timestep=timestep)
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/acme/jax/experiments/run_experiment.py", line 228, in observe
    self._actor.observe(action, next_timestep)
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/acme/agents/jax/actors.py", line 94, in observe
    self._adder.add(
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/acme/adders/reverb/structured.py", line 187, in add
    self._writer.append(
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/reverb/structured_writer.py", line 109, in append
    raise ValueError(
ValueError: Tensor of wrong dtype provided for column 1 (path=('observation', 'action')). Got int32 but expected int64.
[reverb/cc/platform/default/server.cc:84] Shutting down replay server
```

## Error 3

This problem seems to have incorrect error message.
It says `Got double but expected int64` but actually meant by `Got int64 but expected double`.
This may happen when calling `dm_env.restart` with `reward=0` instead of `reward=0.` (to make it a double), this can be demonstrated with `DummyEnv`.


```bash
Traceback (most recent call last):
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/reverb/structured_writer.py", line 94, in append
    self._writer.AppendPartial(flat_data)
ValueError: Tensor of wrong dtype provided for column 2. Got double but expected int64.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/c/Users/rober/OneDrive - Washington University in St. Louis/Documents/lab/RL/SimFish/simfish_env/run_r2d2_simfish.py", line 103, in <module>
    app.run(main)
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/absl/app.py", line 308, in run
    _run_main(main, args)
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/absl/app.py", line 254, in _run_main
    sys.exit(main(argv))
  File "/mnt/c/Users/rober/OneDrive - Washington University in St. Louis/Documents/lab/RL/SimFish/simfish_env/run_r2d2_simfish.py", line 99, in main
    experiments.run_experiment(experiment=config)
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/acme/jax/experiments/run_experiment.py", line 177, in run_experiment
    steps += train_loop.run(num_steps=num_steps)
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/acme/environment_loop.py", line 198, in run
    result = self.run_episode()
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/acme/environment_loop.py", line 122, in run_episode
    self._actor.observe(action, next_timestep=timestep)
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/acme/jax/experiments/run_experiment.py", line 228, in observe
    self._actor.observe(action, next_timestep)
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/acme/agents/jax/actors.py", line 94, in observe
    self._adder.add(
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/acme/adders/reverb/structured.py", line 187, in add
    self._writer.append(
  File "/home/robert/miniconda3/envs/acme/lib/python3.9/site-packages/reverb/structured_writer.py", line 109, in append
    raise ValueError(
ValueError: Tensor of wrong dtype provided for column 2 (path=('observation', 'reward')). Got double but expected int64.
[reverb/cc/platform/default/server.cc:84] Shutting down replay server
```
