TRAIN_ENV=normal
EVAL_ENV=tethered_pred

for seed in {1..8}; do
    python evaluate_r2d2_simfish.py --seed=$seed --dir=/home/asaph/cs_cluster_SAN/SPIM_RAW_Q2_2018/simfish_output/stage2_$TRAIN_ENV --subdir=stage2_${TRAIN_ENV}_$seed --log_subdir=eval_$EVAL_ENV --env_config_file=env_config/$EVAL_ENV.yaml --num_episodes=100 &
done
