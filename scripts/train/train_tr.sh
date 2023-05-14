#!/bin/bash

NUM_GPU=8
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=8
ROOT='.'
method=tr
base_model=bert-base-multilingual-cased
seed=0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path $base_model \
    --data_dir $ROOT/data \
    --dataset_name six-datasets-for-xtreme \
    --output_dir $ROOT/result/$method-$base_model.$seed \
    --max_steps 100000 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model avg_p@1 \
    --load_best_model_at_end \
    --eval_steps 500 \
    --save_steps 2000 \
    --logging_steps 100 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --ams_margin 0. \
    --do_train \
    --seed $seed \
    "$@"
