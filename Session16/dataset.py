#!/usr/bin/env python3
"""
Script to extract and transform the dataset using PyTorch's Dataset class
Author: Shilpaj Bhalerao
Date: Aug 27, 2023
"""
# Third-Party Imports
import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    """
    Class to download and convert data into PyTorch's dataset
    """
    # TODO: Annotation for tokens and dataset
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang: str, tgt_lang: str, seq_len: int):
        """
        Constructor
        :param ds: Dataset to be trained on
        :param tokenizer_src: Tokens for source language
        :param tokenizer_tgt: Tokens for target language
        :param src_lang: Source language
        :param tgt_lang: Target language
        :param seq_len: Maximum number of words in a sentence from the dataset
        """
        # Initialize the Parent Class: Dataset
        super().__init__()

        # Store the data for the instance
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # Create Tokens for Start-Of-Sentence, End-Of-Sentence and Padding
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        """
        Length of the dataset
        """
        return len(self.ds)

    def __getitem__(self, idx: int):
        """
        Method to return data for training
        :param idx: Index of the data sample
        """
        # Extract source and target sentence
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # # Calculate number of words to be padded for encoder and decoder sample
        # enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # Encoder is provided with sos and eos token
        # dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # Decoder is only provided with sos token
        #
        # # Make sure the number of padding token is not negative. If it is, the sentence is too long
        # if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
        #     raise ValueError("Sentence is too long")

        # Add sos and eos token for encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token
            ],
            dim=0,
        )

        # Add only sos to decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64)
            ],
            dim=0,
        )

        # Add only eos token to label. Label allows us to run our decoder into parallel steps
        # This is ground truth for the decoder i.e. decoder will try to generate this
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token
            ],
            dim=0,
        )

        # # Double-check the size of the tensors to ensure they are all seq_len long
        # assert encoder_input.size(0) == self.seq_len
        # assert decoder_input.size(0) == self.seq_len
        # assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,                                                                             # (seq_len)
            "decoder_input": decoder_input,                                                                             # (seq_len)
            # "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),                          # (1, 1, seq_len)
            # "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,                                                                                             # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
            "encoder_token_len": len(encoder_input),
            "decoder_token_len": len(decoder_input),
            "pad_token": self.pad_token
        }


def casual_mask(size: int) -> bool:
    """
    Function to return upper triangular mask of values 1
    :param size: Height/Width of upper triangle
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def collate_batch(batch):
    """
    Function to implement Dynamic Padding
    """
    # Lists to store data
    encoder_input_list = []
    decoder_input_list = []
    decoder_mask_list = []
    encoder_mask_list = []
    label_list = []
    src_text_list = []
    target_text_list = []

    # Max len in a batch
    max_encoder_batch_len = max(x['encoder_token_len'] for x in batch)
    max_decoder_batch_len = max(x['decoder_token_len'] for x in batch)

    for b in batch:
        # Dynamic Padding
        enc_num_padding_tokens = max_encoder_batch_len - len(b['encoder_input'])
        dec_num_padding_tokens = max_decoder_batch_len - len(b['decoder_input'])

        # Make sure the number of padding token is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence too long")

        # Add sos and eos token for encoder input
        encoder_input = torch.cat(
            [
                b['encoder_input'],
                torch.tensor([b['pad_token']] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Add only sos to decoder input
        decoder_input = torch.cat(
            [
                b['decoder_input'],
                torch.tensor([b['pad_token']] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Add only eos token to label. Label allows us to run our decoder into parallel steps
        # This is ground truth for the decoder i.e. decoder will try to generate this
        label = torch.cat(
            [
                b['label'],
                torch.tensor([b['pad_token']] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        encoder_mask = (encoder_input != b['pad_token']).unsqueeze(0).unsqueeze(0).unsqueeze(0).int()  # 1,1,seq_len
        decoder_mask = ((decoder_input != b['pad_token']).unsqueeze(0).int() & casual_mask(decoder_input.size(0))).unsqueeze(0)

        # Append all data
        encoder_input_list.append(encoder_input)
        decoder_input_list.append(decoder_input)
        encoder_mask_list.append(encoder_mask)
        decoder_mask_list.append(decoder_mask)
        label_list.append(label)
        src_text_list.append(b['src_text'])
        target_text_list.append(b['tgt_text'])

        return {
            "encoder_input": torch.vstack(encoder_input_list),
            "decoder_input": torch.vstack(decoder_input_list),
            "encoder_mask": torch.vstack(encoder_mask_list),
            "decoder_mask": torch.vstack(decoder_mask_list),
            "label": torch.vstack(label_list),
            "src_text": src_text_list,
            "tgt_text": target_text_list
        }
