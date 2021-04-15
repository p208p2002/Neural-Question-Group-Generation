from transformers import AutoTokenizer
import torch
from loguru import logger

QUESTION_PREFIX_TOKEN = '[Q:]'
ANSWER_PREFIX_TOKEN = '[A:]'

general_extra_tokens = [QUESTION_PREFIX_TOKEN,ANSWER_PREFIX_TOKEN]

def create_base_tokenizer(base_model,extra_tokens=[]):
    if 'tokenizer' not in globals():
        global tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        # add special token if needed
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info(f"set pad_token to {tokenizer.pad_token}({tokenizer.pad_token_id})")
        if tokenizer.sep_token is None:
            tokenizer.add_special_tokens({'sep_token': '[SEP]'})
            logger.info(f"set sep_token to {tokenizer.sep_token}({tokenizer.sep_token_id})")
        if tokenizer.eos_token is None:
            print('set eos_token...')
            tokenizer.add_special_tokens({'eos_token': '[EOS]'})
            logger.info(f"set eos_token to {tokenizer.eos_token}({tokenizer.eos_token_id})")
        logger.info(f"add general_extra_tokens:{general_extra_tokens}")
        tokenizer.add_tokens(general_extra_tokens,special_tokens=True)
        logger.info(f"add extra_tokens:{extra_tokens}")
        tokenizer.add_tokens(extra_tokens,special_tokens=True)
    return tokenizer