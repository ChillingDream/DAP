This repository contains the code for our paper Dual-Alignment Pre-training for Cross-lingual Sentence Embedding.

The codebase is built upon SimCSE.

## Requirements

* python >= 3.6.10
* pytorch >= 1.7.1

Then run the following script to prepare environment,

```bash
bash setup.sh
```

## Prepare data
Download raw training data
```bash
python data/download_data.py --data_dir [data root path]
```
Filter and pair sentences (this may take several hour)
```
python data/gen_sent_pair.py --name [dataset name] --data_dir [data root path]
```
Generate Enlgish token ids file [Optional]\
This is for accelerating the computation of RTL objective. As the RTL only predicts English tokens, there is no need to compute softmax over the whole vocabulary.
```bash
python data/gen_en_vocab.py \
    --model_name_or_path bert-base-multilingual-cased \
    --data_folder [(data root path)/(dataset name)]
```
Downlod Tatoeba and BUCC
```bash
bash data/download_eval_data.sh [data root path]
```
Note: the gold label of BUCC test set is not publicized. You have to acquire the label by our own and put it under the data folder.

## Evaluation
### Example
```bash
bash scripts/eval/eval_tatoeba.sh
bash scripts/eval/eval_bucc.sh
bash scripts/eval/eval_xnli.sh
```
## Training
### Example
```bash
bash scripts/train/run_rtl_target.sh
```
### Explanation
```bash
python train.py \
    --model_name_or_path [bert-base-multilingual-cased|xlm-roberta-base] \
    --data_dir [data root path] \
    --dataset_name [name of data folder] \
    --en_token_ids_file [the file of a list of the English token ids] (optional) \
    --output_dir [output directory] \
    --max_steps 100000 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model avg_p@1 \
    --load_best_model_at_end \
    --eval_steps 500 \
    --save_steps 2000 \
    --logging_steps 100 \
    --pooler_type cls \
    --mlp_only_train \
    --do_rlm \
    --rlm_pattern 'target' \
    --rlm_probability 1.0 \
    --overwrite_output_dir \
    --temp 0.05 \
    --ams_margin 0. \
    --do_train \
    --seed $seed \
    "$@"
```
The core code of RTL is in core/models.py.