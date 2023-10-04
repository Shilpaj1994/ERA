#!/usr/bin/env python3
"""
Configuration file for Transformer Model training
Author: Shilpaj Bhalerao
Date: Aug 27, 2023
"""
# Standard Library Imports
import os
from pathlib import Path

# Third-Party Imports
from tqdm import tqdm
import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

# Local Imports
from config import get_config, get_weights_file_path
from dataset import BilingualDataset, casual_mask
from model import build_transformer


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """

    :param model:
    :param source:
    :param source_mask:
    :param tokenizer_src:
    :param tokenizer_tgt:
    :param max_len:
    :param device:
    :return:
    """
    # Get sos and eos index
    sos_index = tokenizer_tgt.token_to_id('[SOS]')
    eos_index = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 2).fill_(sos_index).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for target
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([
            decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
        ],
            dim=1
        )

        if next_word == eos_index:
            break
    return decoder_input.squeeze(0)


def get_all_sentences(ds, lang):
    """

    :param ds:
    :param lang:
    :return:
    """
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    """

    :param config:
    :param ds:
    :param lang:
    :return:
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_model(config, vocab_src_len, vocab_tgt_len):
    """

    :param config:
    :param vocab_src_len:
    :param vocab_tgt_len:
    :return:
    """
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])
    return model


# Diff: collate_batch() not present
