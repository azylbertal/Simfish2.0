FROM python:3.9-slim
WORKDIR /simfish
COPY requirements.txt .

RUN pip install -r requirements.txt


ENV PYTHONUNBUFFERED=1
ENV JAX_ENABLE_X64=True

RUN apt-get update && apt-get install -y build-essential cmake

COPY . .

RUN python setup.py build_ext --inplace
ENTRYPOINT ["python", "run_r2d2_simfish.py", "--lp_launch_type=local_mp", "--dir=/output"]
CMD ["--num_actors=3", "--subdir=test", "--seed=1", "--env_config_file=env_config/stage1_env.json"]
