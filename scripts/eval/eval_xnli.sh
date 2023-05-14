NUM_GPU=8
export OMP_NUM_THREADS=8
ROOT='.'
method=rtl
base_model=bert-base-multilingual-cased
seed=0

meta_model_name=$method-$base_model
model_name=$meta_model_name.$seed
model_path=$ROOT/result/$model_name

lr=5e-5
result_path=$ROOT/result/$meta_model_name

python eval_xnli.py \
  --model_name_or_path $model_path \
  --train_language en \
  --do_train \
  --per_device_train_batch_size 256 \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --output_dir $xnli_model_path \
  --overwrite_output_dir \
  --save_steps -1

python eval_xnli.py \
  --model_name_or_path $xnli_model_path \
  --do_eval_all \
  --do_test_all \
  --per_device_eval_batch_size 512 \
  --max_seq_length 128 \
  --output_dir $result_path \
  --seed $seed