from transformers import AutoTokenizer
from .argparser import get_args
import torch
from .config import SEP_TOKEN

def get_tokenizer(args = get_args()):
    if 'tokenizer' not in globals():
        global tokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        # add special token if needed
        if tokenizer.pad_token is None:
            print('set pad_token...')
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if tokenizer.sep_token is None:
            print('set sep_token...')
            tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        if tokenizer.eos_token is None:
            print('set eos_token...')
            tokenizer.add_special_tokens({'eos_token': '[EOS]'})
        tokenizer.add_tokens([SEP_TOKEN],special_tokens=True)

    return tokenizer
