import os
import sys
import zipfile
from urllib import request
from functools import partial
from argparse import ArgumentParser

europarl_lang_list = ['bg', 'cs', 'da', 'de', 'el', 'es', 'et', 'fi', 'fr', 'hu', 'it', 'lt', 'lv', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv']
tanzil_lang_list = ['am', 'ar', 'az', 'bg', 'bn', 'bs', 'cs', 'de', 'dv', 'en', 'es', 'fa', 'fr', 'ha', 'hi', 'id', 'it', 'ja', 'ko', 'ku', 'ml', 'ms', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sd', 'so', 'sq', 'sv', 'sw', 'ta', 'tg', 'th', 'tr', 'tt', 'ug', 'ur', 'uz', 'zh']
opensubtitles_lang_list = ['af', 'ar', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gl', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'ka', 'kk', 'ko', 'lt', 'lv', 'mk', 'ml', 'ms', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi']
unpc_lang_list = ['ar', 'es', 'fr', 'ru', 'zh']
xtreme_lang_list = ['af', 'ar', 'bg', 'bn', 'de', 'el', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'jv', 'ka', 'kk', 'ko', 'ml', 'mr', 'nl', 'pt', 'ru', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ur', 'vi', 'zh']
xnli_lang_list = ['ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
ccmatrix_lang_list = ['af', 'am', 'ar', 'ast', 'az', 'be', 'bg', 'bn', 'br', 'ca', 'ceb', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'ha', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'ig', 'ilo', 'is', 'it', 'ja', 'jv', 'ko', 'la', 'lb', 'lg', 'lt', 'lv', 'mg', 'mk', 'ml', 'mr', 'ms', 'my', 'ne', 'nl', 'no', 'oc', 'om', 'or', 'pl', 'pt', 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh', 'zu']
wikimatrix_lang_list = ['an', 'ar', 'arz', 'as', 'az', 'azb', 'ba', 'bar', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'ceb', 'cs', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'fy', 'gl', 'gom', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'io', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'ko', 'la', 'lb', 'lmo', 'lt', 'mg', 'mk', 'ml', 'mr', 'mwl', 'nds', 'nds_nl', 'ne', 'nl', 'no', 'oc', 'pl', 'pt', 'rm', 'ro', 'ru', 'scn', 'sh', 'si', 'simple', 'sk', 'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'tg', 'tl', 'tr', 'tt', 'ug', 'uk', 'vi', 'wuu', 'zh']

parser = ArgumentParser()
parser.add_argument('--name', type=str, default='six-datasets-1M')
parser.add_argument('--data_dir', type=str, default='/mnt/v-liziheng/data')
args = parser.parse_args()
data_dir = args.data_dir

os.makedirs(os.path.join(data_dir, 'europarl'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'unpc'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'tanzil'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'open-subtitles'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'ccmatrix'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'wikimatrix'), exist_ok=True)

def _progress(block_num, block_size, total_size, filename):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(block_num * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()


def download_url(url, path):
    if not os.path.exists(path):
        try:
            request.urlretrieve(url, path, partial(_progress, filename=url))
        except Exception as e:
            print(e, file=sys.stdout)
            os.remove(path)
            exit(0)
        print(file=sys.stdout)
    else:
        print(f'{path} existed.', file=sys.stdout)


def download_tanzil(lang1, lang2='en'):
    lang_fi = min(lang1, lang2)
    lang_se = max(lang1, lang2)
    url = f'https://object.pouta.csc.fi/OPUS-Tanzil/v1/moses/{lang_fi}-{lang_se}.txt.zip'
    dataset_dir = os.path.join(data_dir, 'tanzil')
    zip_path = os.path.join(dataset_dir, f'{lang_fi}-{lang_se}.zip')
    download_url(url, zip_path)
    zip_file = zipfile.ZipFile(zip_path)
    lang1_path = os.path.join(dataset_dir, f'{lang1}-{lang2}.{lang1}')
    lang2_path = os.path.join(dataset_dir, f'{lang1}-{lang2}.{lang2}')
    if not os.path.exists(lang1_path):
        zip_file.extract(f'Tanzil.{lang_fi}-{lang_se}.{lang1}', dataset_dir)
        os.rename(os.path.join(dataset_dir, f'Tanzil.{lang_fi}-{lang_se}.{lang1}'), lang1_path)
    if not os.path.exists(lang2_path):
        zip_file.extract(f'Tanzil.{lang_fi}-{lang_se}.{lang2}', dataset_dir)
        os.rename(os.path.join(dataset_dir, f'Tanzil.{lang_fi}-{lang_se}.{lang2}'), lang2_path)
    zip_file.close()
    

def download_opensubtitles(lang1, lang2='en'):
    lang_fi = min(lang1, lang2)
    lang_se = max(lang1, lang2)
    url = f'https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/{lang_fi}-{lang_se}.txt.zip'
    dataset_dir = os.path.join(data_dir, 'open-subtitles')
    zip_path = os.path.join(dataset_dir, f'{lang_fi}-{lang_se}.zip')
    download_url(url, zip_path)
    zip_file = zipfile.ZipFile(zip_path)
    lang1_path = os.path.join(dataset_dir, f'{lang1}-{lang2}.{lang1}')
    lang2_path = os.path.join(dataset_dir, f'{lang1}-{lang2}.{lang2}')
    if not os.path.exists(lang1_path):
        zip_file.extract(f'OpenSubtitles.{lang_fi}-{lang_se}.{lang1}', dataset_dir)
        os.rename(os.path.join(dataset_dir, f'OpenSubtitles.{lang_fi}-{lang_se}.{lang1}'), lang1_path)
    if not os.path.exists(lang2_path):
        zip_file.extract(f'OpenSubtitles.{lang_fi}-{lang_se}.{lang2}', dataset_dir)
        os.rename(os.path.join(dataset_dir, f'OpenSubtitles.{lang_fi}-{lang_se}.{lang2}'), lang2_path)
    zip_file.close()


def download_europarl(lang1, lang2='en'):
    lang_fi = min(lang1, lang2)
    lang_se = max(lang1, lang2)
    url = f'https://object.pouta.csc.fi/OPUS-Europarl/v8/moses/{lang_fi}-{lang_se}.txt.zip'
    dataset_dir = os.path.join(data_dir, 'europarl')
    zip_path = os.path.join(dataset_dir, f'{lang_fi}-{lang_se}.zip')
    download_url(url, zip_path)
    zip_file = zipfile.ZipFile(zip_path)
    lang1_path = os.path.join(dataset_dir, f'{lang1}-{lang2}.{lang1}')
    lang2_path = os.path.join(dataset_dir, f'{lang1}-{lang2}.{lang2}')
    if not os.path.exists(lang1_path):
        zip_file.extract(f'Europarl.{lang_fi}-{lang_se}.{lang1}', dataset_dir)
        os.rename(os.path.join(dataset_dir, f'Europarl.{lang_fi}-{lang_se}.{lang1}'), lang1_path)
    if not os.path.exists(lang2_path):
        zip_file.extract(f'Europarl.{lang_fi}-{lang_se}.{lang2}', dataset_dir)
        os.rename(os.path.join(dataset_dir, f'Europarl.{lang_fi}-{lang_se}.{lang2}'), lang2_path)
    zip_file.close()


def download_unpc(lang1, lang2='en'):
    lang_fi = min(lang1, lang2)
    lang_se = max(lang1, lang2)
    url = f'https://object.pouta.csc.fi/OPUS-UNPC/v1.0/moses/{lang_fi}-{lang_se}.txt.zip'
    dataset_dir = os.path.join(data_dir, 'unpc')
    zip_path = os.path.join(dataset_dir, f'{lang_fi}-{lang_se}.zip')
    download_url(url, zip_path)
    zip_file = zipfile.ZipFile(zip_path)
    lang1_path = os.path.join(dataset_dir, f'{lang1}-{lang2}.{lang1}')
    lang2_path = os.path.join(dataset_dir, f'{lang1}-{lang2}.{lang2}')
    if not os.path.exists(lang1_path):
        zip_file.extract(f'UNPC.{lang_fi}-{lang_se}.{lang1}', dataset_dir)
        os.rename(os.path.join(dataset_dir, f'UNPC.{lang_fi}-{lang_se}.{lang1}'), lang1_path)
    if not os.path.exists(lang2_path):
        zip_file.extract(f'UNPC.{lang_fi}-{lang_se}.{lang2}', dataset_dir)
        os.rename(os.path.join(dataset_dir, f'UNPC.{lang_fi}-{lang_se}.{lang2}'), lang2_path)
    zip_file.close()

def download_ccmatrix(lang1, lang2='en'):
    lang_fi = min(lang1, lang2)
    lang_se = max(lang1, lang2)
    url = f'https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/{lang_fi}-{lang_se}.txt.zip'
    dataset_dir = os.path.join(data_dir, 'ccmatrix')
    zip_path = os.path.join(dataset_dir, f'{lang_fi}-{lang_se}.zip')
    download_url(url, zip_path)
    zip_file = zipfile.ZipFile(zip_path)
    lang1_path = os.path.join(dataset_dir, f'{lang1}-{lang2}.{lang1}')
    lang2_path = os.path.join(dataset_dir, f'{lang1}-{lang2}.{lang2}')
    if not os.path.exists(lang1_path):
        zip_file.extract(f'CCMatrix.{lang_fi}-{lang_se}.{lang1}', dataset_dir)
        os.rename(os.path.join(dataset_dir, f'CCMatrix.{lang_fi}-{lang_se}.{lang1}'), lang1_path)
    if not os.path.exists(lang2_path):
        zip_file.extract(f'CCMatrix.{lang_fi}-{lang_se}.{lang2}', dataset_dir)
        os.rename(os.path.join(dataset_dir, f'CCMatrix.{lang_fi}-{lang_se}.{lang2}'), lang2_path)
    zip_file.close()

def download_wikimatrix(lang1, lang2='en'):
    lang_fi = min(lang1, lang2)
    lang_se = max(lang1, lang2)
    url = f'https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/{lang_fi}-{lang_se}.txt.zip'
    dataset_dir = os.path.join(data_dir, 'wikimatrix')
    zip_path = os.path.join(dataset_dir, f'{lang_fi}-{lang_se}.zip')
    download_url(url, zip_path)
    zip_file = zipfile.ZipFile(zip_path)
    lang1_path = os.path.join(dataset_dir, f'{lang1}-{lang2}.{lang1}')
    lang2_path = os.path.join(dataset_dir, f'{lang1}-{lang2}.{lang2}')
    if not os.path.exists(lang1_path):
        zip_file.extract(f'WikiMatrix.{lang_fi}-{lang_se}.{lang1}', dataset_dir)
        os.rename(os.path.join(dataset_dir, f'WikiMatrix.{lang_fi}-{lang_se}.{lang1}'), lang1_path)
    if not os.path.exists(lang2_path):
        zip_file.extract(f'WikiMatrix.{lang_fi}-{lang_se}.{lang2}', dataset_dir)
        os.rename(os.path.join(dataset_dir, f'WikiMatrix.{lang_fi}-{lang_se}.{lang2}'), lang2_path)
    zip_file.close()

for lang in xtreme_lang_list:
    if lang in europarl_lang_list:
        download_europarl(lang)
    if lang in unpc_lang_list:
        download_unpc(lang)
    if lang in tanzil_lang_list:
        download_tanzil(lang)
    if lang in opensubtitles_lang_list:
        download_opensubtitles(lang)
    if lang in ccmatrix_lang_list and lang not in ['ar', 'de', 'es', 'fr', 'it', 'nl', 'pt', 'ru', 'zh']:
        download_ccmatrix(lang)
    if lang in wikimatrix_lang_list:
        download_wikimatrix(lang)