root='.'
method=rtl
seed=0
base_model=bert-base-multilingual-cased
model_name=$method-$base_model
model_path=$root/result/$model_name.$seed

python eval_bucc.py \
    --data_dir $root/data/bucc2018 \
    --model_name_or_path $model_path \
    --max_seq_length 32 \
    --seed $seed \
    --pooler cls_before_pooler \
    --csv_log_dir $root/result/$model_name
