from transformers import AutoTokenizer
from .argparser import get_args

_GENERAL_LEVEL = '[GENERAL]'
_EASY_LEVEL = '[SHALLOW]'
_MIDDLE_LEVEL = '[MEDIUM]'
_HIGH_LEVEL = '[DEEP]'

RACE_BOS = _MIDDLE_LEVEL

def get_tokenizer(args = get_args()):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})    
    tokenizer.add_tokens([_GENERAL_LEVEL,_EASY_LEVEL,_MIDDLE_LEVEL,_HIGH_LEVEL],special_tokens=True)
    return tokenizer
