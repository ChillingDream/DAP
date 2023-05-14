import os
import logging
from datasets import load_dataset, concatenate_datasets

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

class MultiLingualData():
    XTREME_LANGS = ['af', 'ar', 'bg', 'bn', 'de', 'el', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'jv', 'ka', 'kk', 'ko', 'ml', 'mr', 'nl', 'pt', 'ru', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ur', 'vi', 'zh']
    XNLI_LANGS = ['ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    def __init__(self, data_dir, dataset_name, split, tokenizer, data_args):
        cache_dir = os.path.join(data_dir, 'cache')
        os.makedirs(os.path.join(cache_dir, dataset_name, split), exist_ok=True)
        if data_args.languages:
            assert dataset_name == 'laser_tatoeba'
            self.lang_list = data_args.languages
        elif dataset_name == 'six-datasets-for-xnli':
            self.lang_list = self.XNLI_LANGS
        else:
            self.lang_list = self.XTREME_LANGS
        self.tokenizer = tokenizer

        if dataset_name.startswith('six-datasets'):
            data_dir = os.path.join(data_dir, 'six-datasets-1M')
        else:
            data_dir = os.path.join(data_dir, dataset_name)

        if dataset_name == 'laser_tatoeba':
            self.datasets = load_dataset('csv', data_dir=data_dir, data_files={lang: f'{lang}-en.tsv' for lang in self.lang_list}, cache_dir='./cache', delimiter='\t')
        elif dataset_name in ['six-datasets-for-xnli', 'six-datasets-for-xtreme']:
            data_files = {lang: f'{lang}-en.tsv' for lang in self.lang_list}
            self.datasets = load_dataset('csv', data_dir=os.path.join(data_dir, split), data_files=data_files, cache_dir='./cache', delimiter='\t')
        else:
            raise NotImplementedError

        if split == 'train':
            logging.info('Train languages: ' + str(self.lang_list))
        else:
            logging.info('Evaluate languages: ' + str(self.lang_list))
        
        os.makedirs(os.path.join(cache_dir, dataset_name, split, tokenizer.name_or_path), exist_ok=True)
        for lang_id, lang in enumerate(self.lang_list):
            sent0_cname, sent1_cname = self.datasets[lang].column_names

            def prepare_features(examples):
                total = len(examples[sent0_cname])

                # Avoid "None" fields 
                for idx in range(total):
                    if examples[sent0_cname][idx] is None:
                        examples[sent0_cname][idx] = " "
                    if examples[sent1_cname][idx] is None:
                        examples[sent1_cname][idx] = " "
                
                sentences = examples[sent0_cname] + examples[sent1_cname]

                sent_features = tokenizer(
                    sentences,
                    max_length=data_args.max_seq_length,
                    truncation=True,
                    padding=False,
                )
                features = {}
                for key in sent_features:
                    features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]

                langs = [[[lang_id + 2] * data_args.max_seq_length, [1] * data_args.max_seq_length]] * total
                features['langs'] = langs
                    
                return features

            self.datasets[lang] = self.datasets[lang].map(
                prepare_features,
                batched=True,
                remove_columns=self.datasets[lang].column_names,
                load_from_cache_file=True,
                cache_file_name=os.path.join(cache_dir, dataset_name, split, tokenizer.name_or_path, f'{lang}-en.arrow')
            )
            
        if split == 'train':
            self.concat_dataset = concatenate_datasets(list(self.datasets.values()))
