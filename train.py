import logging
import math
import os
import sys
from tkinter.tix import Meter
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
from regex import P
import torch
import collections
import random

from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from core.models import RobertaForRTL, BertForRTL, BertOnlyRTLHead, RobertaRTLHead
from core.trainers import CLTrainer
from data.dataset import MultiLingualData

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    use_lang_emb: bool = field(
        default=False,
        metadata={
            "help": "Will use the language embeddings"
        }
    )
    disable_dropout: bool = field(
        default=False,
        metadata={"help": "If True all dropout will be disabled"}
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    cl_dir: str = field(
        default='fwd',
        metadata={
            "help": "'fwd' means only compute softmax over target language."
            "'bwd' means only compute over source language. 'both' means both."
        }
    )
    ams_margin: float = field(
        default=0.0,
        metadata={
            "help": "The margin used in additive margin softmax"
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_rtl: bool = field(
        default=False,
        metadata={
            "help": "Whether to use RTL auxiliary objective."
        }
    )
    do_masked_rtl: bool = field(
        default=False,
        metadata={
            "help": "Whether to use masked RTL auxiliary objective."
        }
    )
    rtl_pattern: str = field(
        default='source',
        metadata={
            "help": "How to concatenate two sentence representations (only effective if --do_rtl)."
        }
    )
    rtl_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for RTL auxiliary objective (only effective if --do_rtl)."
        }
    )
    do_tlm: bool = field(
        default=False,
        metadata={
            "help": "Whter to use TLM objective."
        }
    )
    tlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for TLM auxiliary objective (only effective if --do_tlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training."
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory of the datasets to use."}
    )
    languages: Optional[List[str]] = field(
        default=None,
        metadata={'help': "Which languages to use."}
    )
    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    en_token_ids_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The English token ids file (.txt)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    rtl_probability: float = field(
        default=1.,
        metadata={
            "help": "Probability of RTL objective"
        }
    )
    tlm_probability: float = field(
        default=0.15,
        metadata={
            "help": "Probability of TLM objective."
        }
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "tsv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.disable_tqdm:
        disable_progress_bar()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    
    if model_args.disable_dropout:
        for k in config.__dict__.keys():
            if "dropout" in k:
                setattr(config, k, 0.0)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    with training_args.main_process_first(desc='load train&dev dataset and map pre-processing'):
        train_dataset = MultiLingualData(data_args.data_dir, data_args.dataset_name, 'train', tokenizer, data_args)
        config.langs = train_dataset.lang_list
        config.n_langs = len(config.langs) + 2
        train_dataset = train_dataset.concat_dataset
        val_datasets = MultiLingualData(data_args.data_dir, data_args.dataset_name, 'dev', tokenizer, data_args).datasets

    config.use_lang_emb = model_args.use_lang_emb
    model_args.mask_token_id = tokenizer.mask_token_id
    model_args.pad_token_id = tokenizer.pad_token_id

    if model_args.model_name_or_path:
        if "roberta" in model_args.model_name_or_path:
            meta_model = RobertaForRTL
        elif "bert" in model_args.model_name_or_path or \
            'LaBSE' in model_args.model_name_or_path:
            meta_model = BertForRTL
        else:
            raise NotImplementedError
            
        model = meta_model.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            model_args=model_args
        )
        if model_args.do_tlm:
            pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
            model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        if model_args.do_rtl:
            en_token_ids = None
            decoder = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path)
            decoder_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            if 'roberta' in model_args.model_name_or_path:
                del decoder.roberta.encoder.layer[:-2]
            else:
                del decoder.bert.encoder.layer[:-2]
            if data_args.en_token_ids_file is not None:
                en_token_ids = []
                with open(os.path.join(data_args.data_dir, data_args.en_token_ids_file)) as f:
                    for line in f:
                        en_token_ids.append(int(line))
                en_token_dict = dict(zip(en_token_ids, range(len(en_token_ids))))
                if 'roberta' in model_args.model_name_or_path:
                    rtl_head = RobertaRTLHead(decoder_config)
                    rtl_head.load_state_dict(decoder.lm_head.state_dict())
                    decoder.lm_head = rtl_head
                else:
                    rtl_head = BertOnlyRTLHead(decoder_config)
                    rtl_head.load_state_dict(decoder.cls.state_dict())
                    decoder.cls = rtl_head
                rtl_head.shrink(en_token_ids)
                decoder.config.vocab_size = len(en_token_ids)
            model.decoder = decoder
    else:
        raise NotImplementedError

    model.resize_token_embeddings(len(tokenizer))

    # Data collator
    @dataclass
    class OurDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        tlm_probability: float = data_args.tlm_probability
        rtl_probability: float = data_args.rtl_probability
        training = True

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            special_keys = ['input_ids', 'langs', 'attention_mask', 'token_type_ids', 'rtl_labels', 'tlm_input_ids', 'tlm_labels']
            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return
            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            if model_args.do_tlm and self.training:
                batch["tlm_input_ids"], batch["tlm_labels"], _ = self.mask_tokens(
                    batch["input_ids"], mask_probability=self.tlm_probability, mask_ratio=0.8, replace_ratio=0.1)

            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

            if model_args.do_rtl and self.training:
                if model_args.rtl_pattern in ['target', 'both', 'TLM']:
                    if self.rtl_probability < 1.:
                        _, tgt_rtl_labels, tgt_rtl_mask = self.mask_tokens(
                            batch["input_ids"][:, 1],
                            special_tokens_mask=False,
                            mask_probability=self.rtl_probability, mask_ratio=1.)
                    else:
                        tgt_rtl_labels = batch["input_ids"][:, 1].clone()
                        tgt_rtl_mask = torch.full_like(tgt_rtl_labels, True, dtype=torch.bool)
                if model_args.rtl_pattern in ['source', 'both', 'TLM']:
                    if self.rtl_probability < 1.:
                        _, src_rtl_labels, src_rtl_mask = self.mask_tokens(
                            batch["input_ids"][:, 0],
                            special_tokens_mask=False,
                            mask_probability=self.rtl_probability, mask_ratio=1.)
                    else:
                        src_rtl_labels = batch["input_ids"][:, 0].clone()
                        src_rtl_mask = torch.full_like(src_rtl_labels, True, dtype=torch.bool)

                if model_args.rtl_pattern in ['both', 'TLM']:
                    rtl_mask = (src_rtl_mask, tgt_rtl_mask)
                elif model_args.rtl_pattern == 'target':
                    rtl_mask = (None, tgt_rtl_mask)
                    src_rtl_labels = torch.full_like(batch["input_ids"][:, 0], -100)
                elif model_args.rtl_pattern == 'source':
                    rtl_mask = (src_rtl_mask, None)
                    tgt_rtl_labels = torch.full_like(batch["input_ids"][:, 1], -100)
                batch["rtl_mask"] = rtl_mask
                if en_token_ids is not None:
                    assert model_args.rtl_pattern == 'target'
                    for i in range(tgt_rtl_labels.size(0)):
                        for j in range(tgt_rtl_labels.size(1)):
                            tgt_rtl_labels[i, j] = en_token_dict[tgt_rtl_labels[i, j].item()] if tgt_rtl_labels[i, j] >= 0 else -100
                rtl_labels = torch.cat([src_rtl_labels, tgt_rtl_labels], dim=1)
                rtl_labels[rtl_labels == tokenizer.pad_token_id] = -100
                assert len(rtl_labels.shape) == 2
                batch["rtl_labels"] = rtl_labels
            

            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch
        
        def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None,
            mask_probability: float = 0.2, mask_ratio: float = 1., replace_ratio: float = 0.
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: mask_ratio MASK, replace_ratio random, 1-mask_ratio-replace_ratio original.
            """
            assert mask_ratio + replace_ratio <= 1
            inputs = inputs.clone()
            labels = inputs.clone()
            # We sample a few tokens in each sequence for masking (with probability `mask_probability`)
            probability_matrix = torch.full(labels.shape, mask_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            elif special_tokens_mask != False:
                special_tokens_mask = special_tokens_mask.bool()

            if special_tokens_mask != False:
                probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # mask_ratio of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, mask_ratio)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            if replace_ratio > 0:
                # replace_ratio of the time, we replace masked input tokens with random word
                indices_random = torch.bernoulli(torch.full(labels.shape, replace_ratio / (1 - mask_ratio))).bool() & masked_indices & ~indices_replaced
                random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
                inputs[indices_random] = random_words[indices_random]

            # The rest of the time (1 - mask_ratio - replace_ratio) of the time) we keep the masked input tokens unchanged
            return inputs, labels, masked_indices

    data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.model_args = model_args
    trainer.validation_datasets = val_datasets

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=True)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
