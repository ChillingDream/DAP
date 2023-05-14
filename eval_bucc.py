import sys
import io, os
import numpy as np
import logging
import argparse
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, DataCollatorWithPadding
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from datasets import load_dataset
from core.models import XLMBertModel

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

class OurModel:
    def __init__(self, model, pooler, device, layer_id):
        self.model = model.to(device)
        self.model.eval()
        self.pooler = pooler
        self.device = device
        self.layer_id = layer_id
    
    def __call__(self, batch):
        bs = batch['input_ids'].size(0)
        batch = {k: w.to(self.device) for k, w in batch.items()}
        outputs = self.model(**batch, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states
        if self.layer_id is not None:
            last_hidden = outputs.hidden_states[self.layer_id]

        # Apply different poolers
        if self.pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            return pooler_output
        elif self.pooler == 'cls_before_pooler':
            return last_hidden[:, 0]
        elif self.pooler == "avg":
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1))
        else:
            raise NotImplementedError


def extract_embeddings(model, tokenizer, src_lang, tgt_lang, split, args):
    bucc_datasets = load_dataset(
        'csv',
        data_dir=args.data_dir,
        data_files={
            src_lang: f'{src_lang}-{tgt_lang}.{split}.{src_lang}',
            tgt_lang: f'{src_lang}-{tgt_lang}.{split}.{tgt_lang}'
        },
        delimiter='\t',
        names=['id', 'text']
    )
    def process(examples):
        return tokenizer(
            examples['text'],
            max_length=args.max_seq_length,
            truncation=True,
            padding=False
        )
    ret = []
    for ds in bucc_datasets.values():
        ret.append(ds['id'])
        ds = ds.map(
            process,
            load_from_cache_file=True,
            batched=True,
            remove_columns=ds.column_names
        )
        dataloader = DataLoader(ds, collate_fn=DataCollatorWithPadding(tokenizer), batch_size=args.batch_size)
        embeddings = []
        for batch in tqdm(dataloader):
            with torch.no_grad():
                embeddings.append(model(batch).cpu())
        embeddings = torch.cat(embeddings, dim=0)
        ret.append(embeddings)
    return ret


def knn(x, y, k, batch_size):
    assert k <= len(y)
    logging.info(' - finding {:d}-nn among {:d} candidates using'.format(k, y.shape[0]))
    sim = []
    ind = []
    y = y.cuda()
    for i in trange(0, len(x), batch_size):
        bsim = []
        x_batch = x[i: i + batch_size].cuda()
        for j in range(0, len(y), batch_size):
            y_batch = y[j: j + batch_size]
            bsim.append(F.cosine_similarity(x_batch.unsqueeze(1), y_batch.unsqueeze(0), dim=-1))
        bsim = torch.cat(bsim, dim=1)
        bsim_topk = bsim.topk(k, dim=1)
        sim.append(bsim_topk[0])
        ind.append(bsim_topk[1])
    sim = torch.cat(sim, dim=0).cpu()
    ind = torch.cat(ind, dim=0).cpu()
    return sim, ind


def score(x, y, fwd_mean, bwd_mean, margin, dist='cosine'):
    if dist == 'cosine':
        return margin(F.cosine_similarity(x, y, dim=0), (fwd_mean + bwd_mean) / 2)
    else:
        l2 = ((x - y) ** 2).sum()
        sim = 1 / (1 + l2)
        return margin(sim, (fwd_mean + bwd_mean) / 2)


def score_candidates(x, y, candidate_inds, fwd_mean, bwd_mean, margin, dist='cosine'):
    logging.info(' - scoring {:d} candidates using {}'.format(x.shape[0], dist))
    scores = torch.zeros(candidate_inds.shape)
    for i in trange(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = candidate_inds[i, j]
            scores[i, j] = score(x[i], y[k], fwd_mean[i], bwd_mean[k], margin, dist)
    return scores


def mine_bitext(x, x_inds, y, y_inds, neighborhood, batch_size, threshold=0):
    x2y_sim, x2y_ind = knn(x, y, neighborhood, batch_size)
    x2y_mean = x2y_sim.mean(dim=1)
    y2x_sim, y2x_ind = knn(y, x, neighborhood, batch_size)
    y2x_mean = y2x_sim.mean(dim=1)
    margin = lambda x, y: x / y
    fwd_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin)
    bwd_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, margin)
    fwd_best = x2y_ind[torch.arange(x.shape[0]), fwd_scores.argmax(dim=1)].cpu()
    bwd_best = y2x_ind[torch.arange(y.shape[0]), bwd_scores.argmax(dim=1)].cpu()

    indices = torch.stack((
        torch.cat((torch.arange(x.shape[0]), bwd_best)),
        torch.cat((fwd_best, torch.arange(y.shape[0])))),
        dim=1
    ).tolist()
    scores = torch.cat((fwd_scores.max(dim=1)[0], bwd_scores.max(dim=1)[0]))
    seen_src, seen_tgt = set(), set()
    ret = []

    logging.info('- mining using max retrieval')
    for i in tqdm(scores.argsort(dim=0, descending=True)):
        src_ind, tgt_ind = indices[i]
        if src_ind not in seen_src and tgt_ind not in seen_tgt:
            seen_src.add(src_ind)
            seen_tgt.add(tgt_ind)
            if scores[i] > threshold:
                ret.append(((x_inds[src_ind], y_inds[tgt_ind]), scores[i]))
    cnt = 0
    for x1, y1 in ret:
        for x2, y2 in ret:
            if x1 == x2:
                cnt += 1
    return ret


def bucc_optimize(bitext_pairs, gold):
    ngold = len(gold)
    nextract = ncorrect = 0
    threshold = 0
    best_f1 = 0
    best_prec = 0
    best_recall = 0
    for i in range(len(bitext_pairs)):
        nextract += 1
        if '\t'.join(bitext_pairs[i][0]) in gold:
            ncorrect += 1
        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / ngold
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_prec = precision
                best_recall = recall
                threshold = (bitext_pairs[i][1] + bitext_pairs[i + 1][1]) / 2
    return threshold, best_f1, best_prec, best_recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            help="Transformers' model name or path")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--languages", type=str, default=['fr', 'de', 'ru', 'zh'], nargs='*')
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--neighborhood", type=int, default=4)
    parser.add_argument("--seed", type=int, 
            help="Seed used in training")
    parser.add_argument("--pooler", type=str, 
            choices=['cls', 'cls_before_pooler', 'avg'], 
            default='cls', 
            help="Which pooler to use")
    parser.add_argument("--layer_id", type=int, default=None,
                        help="Which layer's feature to use as the sentence representation")
    parser.add_argument("--csv_log_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    # Load transformers' model checkpoint
    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = OurModel(model, args.pooler, device, args.layer_id)
    for lang in args.languages:
        best_threshold = 0
        for split in ['dev', 'test']:
            x_inds, x, y_inds, y = extract_embeddings(model, tokenizer, lang, 'en', split, args)
            bitext_pairs = mine_bitext(x, x_inds, y, y_inds, args.neighborhood, args.batch_size, best_threshold)
            with open(os.path.join(args.data_dir, f'{lang}-en.{split}.gold')) as f:
                gold = [line.strip() for line in f]
            if split == 'dev':
                best_threshold, f1, prec, recall = bucc_optimize(bitext_pairs, gold)
                logging.info(f'best threshold: {best_threshold}. best_f1: {f1}')
            else:
                ncorrect = 0
                for pair, _ in bitext_pairs:
                    if '\t'.join(pair) in gold:
                        ncorrect += 1
                prec = ncorrect / len(bitext_pairs)
                recall = ncorrect / len(gold)
                f1 = 2 * prec * recall / (prec + recall)
                logging.info(f'prec: {prec}, recall: {recall}, f1: {f1}')

            if args.csv_log_dir is not None:
                os.makedirs(args.csv_log_dir, exist_ok=True)
                output_file_path = os.path.join(args.csv_log_dir, f'bucc_{split}.csv')
                if not os.path.exists(output_file_path):
                    df = pd.DataFrame()
                else:
                    df = pd.read_csv(output_file_path, index_col=[0])
                df.loc[args.seed, lang + '_prec'] = prec * 100
                df.loc[args.seed, lang + '_recall'] = recall * 100
                df.loc[args.seed, lang + '_f1'] = f1 * 100
                df.to_csv(output_file_path)
        

if __name__ == "__main__":
    main()
