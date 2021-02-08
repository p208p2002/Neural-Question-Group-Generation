from transformers import AutoTokenizer
from .argparser import get_args

def get_tokenizer(args = get_args()):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    # tokenizer.eos_token_id = tokenizer.bos_token_id
    return tokenizer
