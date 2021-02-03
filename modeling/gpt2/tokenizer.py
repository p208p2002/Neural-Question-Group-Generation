from transformers import AutoTokenizer
from .argparser import get_args
args = get_args()
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    return tokenizer
