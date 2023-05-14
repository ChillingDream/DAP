import os
import re
import datasets
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from argparse import ArgumentParser

europarl_lang_list = ['bg', 'cs', 'da', 'de', 'el', 'es', 'et', 'fi', 'fr', 'hu', 'it', 'lt', 'lv', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv']
tanzil_lang_list = ['am', 'ar', 'az', 'bg', 'bn', 'bs', 'cs', 'de', 'dv', 'en', 'es', 'fa', 'fr', 'ha', 'hi', 'id', 'it', 'ja', 'ko', 'ku', 'ml', 'ms', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sd', 'so', 'sq', 'sv', 'sw', 'ta', 'tg', 'th', 'tr', 'tt', 'ug', 'ur', 'uz', 'zh']
opensubtitles_lang_list = ['af', 'ar', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gl', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'ka', 'kk', 'ko', 'lt', 'lv', 'mk', 'ml', 'ms', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi']
unpc_lang_list = ['ar', 'es', 'fr', 'ru', 'zh']
xtreme_lang_list = ['af', 'ar', 'bg', 'bn', 'de', 'el', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'jv', 'ka', 'kk', 'ko', 'ml', 'mr', 'nl', 'pt', 'ru', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ur', 'vi', 'zh']
xnli_lang_list = ['ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
ccmatrix_lang_list = ['af', 'am', 'ast', 'az', 'be', 'bg', 'bn', 'br', 'ca', 'ceb', 'cs', 'cy', 'da', 'el', 'en', 'eo', 'et', 'eu', 'fa', 'fi', 'fy', 'ga', 'gd', 'gl', 'ha', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'ig', 'ilo', 'is', 'ja', 'jv', 'ko', 'la', 'lb', 'lg', 'lt', 'lv', 'mg', 'mk', 'ml', 'mr', 'ms', 'my', 'ne', 'no', 'oc', 'om', 'or', 'pl', 'ro', 'sd', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zu']
wikimatrix_lang_list = ['an', 'ar', 'arz', 'as', 'az', 'azb', 'ba', 'bar', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'ceb', 'cs', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'fy', 'gl', 'gom', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'io', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'ko', 'la', 'lb', 'lmo', 'lt', 'mg', 'mk', 'ml', 'mr', 'mwl', 'nds', 'nds_nl', 'ne', 'nl', 'no', 'oc', 'pl', 'pt', 'rm', 'ro', 'ru', 'scn', 'sh', 'si', 'simple', 'sk', 'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'tg', 'tl', 'tr', 'tt', 'ug', 'uk', 'vi', 'wuu', 'zh']
source_dict = {
    'europarl': europarl_lang_list,
    'tanzil': tanzil_lang_list,
    'open-subtitles': opensubtitles_lang_list,
    'unpc': unpc_lang_list,
    'ccmatrix': ccmatrix_lang_list,
    'wikimatrix': wikimatrix_lang_list
}

parser = ArgumentParser()
parser.add_argument('--name', type=str, default='six-datasets-1M')
parser.add_argument('--data_dir', type=str, default='./dataset')
args = parser.parse_args()

full_train_size = 1000000
full_dev_size = 10000
name = args.name
data_dir = args.data_dir
target_dir = os.path.join(data_dir, name)
target_filename = os.path.join(target_dir, f'{name}.tsv')
center_lang = 'en'
n_worker = 32
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenizer.model_max_length = 100000

os.makedirs(os.path.join(target_dir, 'dev'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'train'), exist_ok=True)

def map_function(x):
    return {k: re.sub(r'[\x00-\x08\x0a-\x1f\x7f]', '', w).replace('\t', ' ').replace('"', '') for k, w in x.items()}


def filter_function(x):
    line1, line2 = x['source'], x[center_lang]
    if '@' in line1 or '@' in line2 or 'http' in line1 or 'http' in line2:
        return False 
    if line1 == 'N/A' or line2 == 'N/A':
        return False
    group1 = tokenizer.tokenize(line1)
    group2 = tokenizer.tokenize(line2)
    l1 = len(group1)
    l2 = len(group2)

    if l1 <= 3 or l2 <= 3:
        return False
    if max(l1, l2) > max(min(l1, l2) * 2, 32):
        return False
    if min(l1, l2) > 64:
        return False

    if line1 == line2:
        return False
    
    if l1 <= 5 or l2 <= 5:
        return True

    cnt = 0
    for w1 in group1:
        if w1 in group2:
            cnt += 1
    if cnt * 2 >= l1:
        return False

    cnt = 0
    for w2 in group2:
        if w2 in group1:
            cnt += 1
    if cnt * 2 >= l2:
        return False
    return True



def par_extract(fname1, fname2, fdev_name, target_lang):
    src_texts = load_dataset('text', data_files=fname1).rename_column('text', 'source')['train']
    tgt_texts = load_dataset('text', data_files=fname2).rename_column('text', target_lang)['train']
    data = concatenate_datasets([src_texts, tgt_texts], axis=1)
    data = data.map(map_function, num_proc=n_worker)
    data = data.filter(filter_function, num_proc=n_worker)
    data = datasets.Dataset.from_pandas(data.to_pandas().drop_duplicates(), preserve_index=False)
    data = data.shuffle(42)
    dev_size = min(len(data) // 10, full_dev_size)
    train_size = min(len(data) - dev_size, full_train_size)
    if fdev_name is not None:
        dev_data = data.select(range(dev_size))
        train_data = data.select(range(dev_size, dev_size + train_size))
        dev_data.to_csv(fdev_name, sep='\t', index=False)
        return train_data
    else:
        return data.select(range(min(dev_size + train_size, len(data))))


def gen_pair(data_dir, tgt_lang, mix=True):
    results = []
    for src_lang in xtreme_lang_list:
        lang1 = src_lang
        lang2 = tgt_lang
        data = []
        for src_ds, lang_list in source_dict.items():
            if src_lang in lang_list:
                dataset_dir = os.path.join(data_dir, src_ds)
                fname1 = os.path.join(dataset_dir, f'{lang1}-{lang2}.{lang1}')
                fname2 = os.path.join(dataset_dir, f'{lang1}-{lang2}.{lang2}')
                data.append(par_extract(fname1, fname2, None, tgt_lang))
        if len(data) == 0:
            continue
        data = concatenate_datasets(data, axis=0)
        dev_size = min(len(data) // 10, full_dev_size)
        train_size = min(len(data) - dev_size, full_train_size)
        data = data.shuffle(42)
        dev_data = data.select(range(dev_size))
        train_data = data.select(range(dev_size, min(dev_size + train_size, len(data))))
        fdev_name = os.path.join(target_dir, 'dev', f'{lang1}-{lang2}.tsv')
        dev_data.to_csv(fdev_name, sep='\t', index=False)
        results.append(train_data)
        if not mix:
            results[-1].to_csv(os.path.join(target_dir, 'train', f'{src_lang}-{tgt_lang}.tsv'), sep='\t', index=False)
        print(f'Number of {src_lang}-{tgt_lang} pairs:', len(results[-1]))

    if mix == True:
        return concatenate_datasets(results, axis=0)
    else:
        return results

data = gen_pair(data_dir, center_lang, mix=False)
data.to_csv(target_filename, num_proc=n_worker, sep='\t', index=False)