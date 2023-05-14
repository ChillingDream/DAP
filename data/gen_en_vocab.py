import os
from glob import glob
from tqdm import tqdm, trange
from transformers import AutoTokenizer, BertTokenizer
from datasets import load_dataset
from argparse import ArgumentParser
import pandas as pd


parser = ArgumentParser()
parser.add_argument('--model_name_or_path', type=str)
parser.add_argument('--data_folder', type=str)
parser.add_argument('--train_file', type=str)
parser.add_argument('--num_workers', type=int, default=32)

args = parser.parse_args()

datasets = load_dataset('csv', data_files=glob(os.path.join(args.data_folder, '*', '??-en.tsv')), cache_dir="./cache/", delimiter="\t")
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
tokenizer.model_max_length = 2048
column_names = datasets["train"].column_names
sent0_name, sent1_name = column_names

def process(batch):
    local_vocab = set()
    for x in batch['en']:
        for token in tokenizer.encode(x, truncation=False):
            local_vocab.add(token)
    return {'vocab': list(local_vocab)}

vocab = datasets["train"].map(process, num_proc=args.num_workers, batched=True, batch_size=100000, remove_columns=column_names)
vocab = set(vocab['vocab'])
vocab.add(tokenizer.pad_token)
vocab.add(tokenizer.unk_token)
vocab = sorted(list(vocab))
with open(os.path.join(args.data_folder, 'en_token_ids.txt'), 'w') as f:
    for token in vocab:
        print(token, file=f)