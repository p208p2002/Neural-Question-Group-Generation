from transformers import AutoTokenizer
from .argparser import get_args

_GENERAL_LEVEL = '[GENERAL]'
_EASY_LEVEL = '[SHALLOW]'
_MIDDLE_LEVEL = '[MEDIUM]'
_HIGH_LEVEL = '[DEEP]'

RACE_BOS = _MIDDLE_LEVEL

def get_tokenizer(args = get_args()):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # add special token if needed
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    tokenizer.add_tokens([_GENERAL_LEVEL,_EASY_LEVEL,_MIDDLE_LEVEL,_HIGH_LEVEL],special_tokens=True)
    return tokenizer
