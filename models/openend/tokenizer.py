from .argparser import get_args

from utils.tokenizer import create_base_tokenizer
from .argparser import get_args
from .config import SEP_TOKEN

def get_tokenizer(args = get_args()):
    tokenizer = create_base_tokenizer(args.base_model,extra_tokens=[SEP_TOKEN])
    return tokenizer