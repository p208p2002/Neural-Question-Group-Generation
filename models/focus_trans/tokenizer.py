from utils.tokenizer import create_base_tokenizer
from .argparser import get_args

GENED_TOKEN = '[GEN]'
def get_tokenizer(args = get_args()):
    tokenizer = create_base_tokenizer(args.base_model,extra_tokens=[GENED_TOKEN])
    return tokenizer