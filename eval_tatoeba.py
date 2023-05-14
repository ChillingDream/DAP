import sys
import io, os
import numpy as np
import logging
import argparse
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass, field
#from prettytable import PrettyTable
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from data.dataset import MultiLingualData
from core.models import XLMBertModel

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

class OurModel:
    def __init__(self, model, pooler, device, layer_id=None):
        self.model = model.to(device)
        self.model.eval()
        self.pooler = pooler
        self.device = device
        self.layer_id = layer_id
        self.record = None
    
    def __call__(self, batch):
        bs = batch['input_ids'].size(0)
        batch = {k: w.view(bs * 2, -1).to(self.device) for k, w in batch.items()}
        outputs = self.model(**batch, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states
        if self.layer_id is not None:
            last_hidden = outputs.hidden_states[self.layer_id]
        
        if self.record is not None:
            for i in range(bs):
                token_hiddens = []
                for j in range(1, batch['input_ids'].size(1)):
                    if j == batch['input_ids'].size(1) - 1 or batch['attention_mask'][i][j + 1] == 0:
                        break
                    token_hiddens.append(last_hidden[i][j])
                self.record[0].append(token_hiddens)

                token_hiddens = []
                for j in range(1, batch['input_ids'].size(1)):
                    if j == batch['input_ids'].size(1) - 1 or batch['attention_mask'][i + bs][j + 1] == 0:
                        break
                    token_hiddens.append(last_hidden[i + bs][j])
                self.record[1].append(token_hiddens)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            help="Transformers' model name or path")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--languages", type=str, default=None, nargs='*')
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
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
    model = OurModel(model, args.pooler, device, layer_id=args.layer_id)
    dataset = MultiLingualData(args.data_dir, 'laser_tatoeba', 'test', tokenizer, args)

    @dataclass
    class OurDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = args.max_seq_length
        pad_to_multiple_of: Optional[int] = None

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return
            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] for k in feature if k in special_keys})

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch
        
    metrics = {}
    for lang, ds in dataset.datasets.items():
        print('Evaluating', lang)
        dataloader = DataLoader(ds, batch_size=args.batch_size, collate_fn=OurDataCollatorWithPadding(tokenizer))
        z1_list, z2_list = [], []
        for batch in dataloader:
            with torch.no_grad():
                embeddings = model(batch)
            embeddings = embeddings.view(batch['input_ids'].size(0), 2, -1)
            z1, z2 = embeddings[:, 0], embeddings[:, 1]
            z1_list.append(z1)
            z2_list.append(z2)
        z1_all = torch.cat(z1_list, dim=0).unsqueeze(0)
        z2_all = torch.cat(z2_list, dim=0).unsqueeze(0)

        labels = torch.arange(z1_all.size(1)).long().to(z1.device)
        fwd_pred = []
        bwd_pred = []
        for z1 in z1_list:
            fwd_pred.append(F.cosine_similarity(z1.unsqueeze(1), z2_all, dim=-1).argmax(1))
        for z2 in z2_list:
            bwd_pred.append(F.cosine_similarity(z2.unsqueeze(1), z1_all, dim=-1).argmax(1))
        fwd_pred = torch.cat(fwd_pred, 0)
        bwd_pred = torch.cat(bwd_pred, 0)
        assert fwd_pred.size() == labels.size() and bwd_pred.size() == labels.size()
        fwd_corr = (fwd_pred == labels).sum().item()
        bwd_corr = (bwd_pred == labels).sum().item()
        metrics.update({f'{lang}-en_p@1': fwd_corr / len(labels), f'en-{lang}_p@1': bwd_corr / len(labels)})
    metrics['eval_avg_p@1'] = sum(metrics.values()) / len(metrics)
    logging.info(metrics)

    suffix = '' if args.layer_id is None else f'_{args.layer_id}'
    if args.csv_log_dir is not None:
        os.makedirs(args.csv_log_dir, exist_ok=True)
        output_file_path = os.path.join(args.csv_log_dir, f'tatoeba_xx-en{suffix}.csv')
        if not os.path.exists(output_file_path):
            df = pd.DataFrame()
        else:
            df = pd.read_csv(output_file_path, index_col=[0])
        for lang in dataset.lang_list:
            df.loc[args.seed, lang] = metrics[f'{lang}-en_p@1'] * 100
        df.loc[args.seed, 'avg'] = None
        df.loc[args.seed, 'avg'] = df.loc[args.seed].mean()
        df.to_csv(output_file_path)

        output_file_path = os.path.join(args.csv_log_dir, f'tatoeba_en-xx{suffix}.csv')
        if not os.path.exists(output_file_path):
            df = pd.DataFrame()
        else:
            df = pd.read_csv(output_file_path, index_col=[0])
        for lang in dataset.lang_list:
            df.loc[args.seed, lang] = metrics[f'en-{lang}_p@1'] * 100
        df.loc[args.seed, 'avg'] = None
        df.loc[args.seed, 'avg'] = df.loc[args.seed].mean()
        df.to_csv(output_file_path)


if __name__ == "__main__":
    main()
