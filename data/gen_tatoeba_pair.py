# data comes from xtreme

import os
from argparse import ArgumentParser

lang_dict = {'ara':'ar', 'heb':'he', 'vie':'vi', 'ind':'id',
    'jav':'jv', 'tgl':'tl', 'eus':'eu', 'mal':'ml', 'tam':'ta',
    'tel':'te', 'afr':'af', 'nld':'nl', 'eng':'en', 'deu':'de',
    'ell':'el', 'ben':'bn', 'hin':'hi', 'mar':'mr', 'urd':'ur',
    'tam':'ta', 'fra':'fr', 'ita':'it', 'por':'pt', 'spa':'es',
    'bul':'bg', 'rus':'ru', 'jpn':'ja', 'kat':'ka', 'kor':'ko',
    'tha':'th', 'swh':'sw', 'cmn':'zh', 'kaz':'kk', 'tur':'tr',
    'est':'et', 'fin':'fi', 'hun':'hu', 'pes':'fa', 'aze':'az',
    'lit':'lt', 'pol':'pl', 'ukr':'uk', 'ron':'ro', 'ces':'cs',
    'dan':'da', 'lvs':'lv', 'slk':'sk', 'slv':'sl', 'swe':'sv'}

parser = ArgumentParser()
parser.add_argument('--src_dir', type=str)
parser.add_argument('--tgt_dir', type=str)
args = parser.parse_args()
if not os.path.exists(args.tgt_dir):
    os.mkdir(args.tgt_dir)

def gen_pair(raw_data_dir, target_data_dir, lang1, lang2):
    num = 0
    l1 = lang_dict[lang1]
    l2 = lang_dict[lang2]
    with open(os.path.join(raw_data_dir, f'tatoeba.{lang1}-{lang2}.{lang1}'), 'r') as f1, \
         open(os.path.join(raw_data_dir, f'tatoeba.{lang1}-{lang2}.{lang2}'), 'r') as f2, \
         open(os.path.join(target_data_dir, f'{l1}-{l2}.tsv'), 'w') as f3:
         print(f'source\t{l2}', file=f3)
         for line1, line2 in zip(f1, f2):
            line1 = line1.strip().replace('\t', ' ')
            line2 = line2.strip().replace('\t', ' ')
            line1 = line1.replace('"', '')
            line2 = line2.replace('"', '')
            print(line1 + '\t' + line2, file=f3)
            num += 1
    print(f'Number of {l1}-{l2} pairs: {num}')

for lang1 in lang_dict:
    if lang1 != 'eng':
        gen_pair(args.src_dir, args.tgt_dir, lang1, 'eng')