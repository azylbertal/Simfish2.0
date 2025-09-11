FROM python:3.9-slim
WORKDIR /simfish
COPY requirements_thin.txt .

RUN pip install -r requirements_thin.txt


ENV PYTHONUNBUFFERED=1
ENV JAX_ENABLE_X64=True

RUN apt-get update && apt-get install -y build-essential cmake

COPY . .

RUN python setup.py build_ext --inplace

CMD ["python", "run_r2d2_simfish.py", "--lp_launch_type=local_mp", "--num_actors=3", "--directory=/output/test_sa", "--seed=1", "--num_steps=1_000_000"]
