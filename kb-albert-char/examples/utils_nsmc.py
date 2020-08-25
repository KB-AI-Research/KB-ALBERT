import os
import csv
import logging
import torch
from torch.utils.data.dataset import Dataset
from enum import Enum
from typing import List, Optional, Union
from transformers import InputExample, InputFeatures, PreTrainedTokenizer


logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class NsmcDataset(Dataset):

    features: List[InputFeatures]
    pad_token_label_id: int = torch.nn.CrossEntropyLoss().ignore_index

    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            max_seq_len: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        cached_features_file = os.path.join(
            data_dir, "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_len)),
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            self.features = torch.load(cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {data_dir}")
            examples = read_examples_from_file(data_dir, mode)
            # TODO clean up all this to leverage built-in features of tokenizers
            self.features = convert_examples_to_features(
                examples,
                max_seq_len,
                tokenizer,
            )
            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def read_examples_from_file(data_dir, mode):
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"ratings_{mode}.txt")
    examples = []
    with open(file_path, "r", encoding="utf-8") as r:
        reader = csv.reader(r, delimiter="\t")
        next(reader, None)
        for guid_idx, line in enumerate(reader):
            examples.append(InputExample(guid=f"{mode}-{guid_idx}", text_a=line[1], label=int(line[2])))
    return examples


def convert_examples_to_features(
        examples: List[InputExample],
        max_seq_len: int,
        tokenizer: PreTrainedTokenizer,
        cls_token_segment_id=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`"""
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text_a)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        label_id = example.label

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label_id
            )
        )

    return features


def get_label():
    '''Sentiment analysis negative/positive'''
    return ['negative', 'positive']
    # return [0, 1]