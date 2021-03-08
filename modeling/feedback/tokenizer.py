from transformers import AutoTokenizer
from .argparser import get_args
import torch

# _GENERAL_LEVEL = '_$[GENERAL]'
# _EASY_LEVEL = '_$[SHALLOW]'
# _MIDDLE_LEVEL = '_$[MEDIUM]'
# _HIGH_LEVEL = '_$[DEEP]'

# RACE_BOS = _MIDDLE_LEVEL

def get_tokenizer(args = get_args()):
    if 'tokenizer' not in globals():
        global tokenizer

        if args.base_model == 'google/reformer-enwik8':
            tokenizer = ReformerTokenizer()
        else:
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
            # tokenizer.add_tokens([_GENERAL_LEVEL,_EASY_LEVEL,_MIDDLE_LEVEL,_HIGH_LEVEL],special_tokens=True)

    return tokenizer

class ReformerTokenizer():
    def __init__(self):
        self.pad_token_id = 0
        self.sep_token = '[SEP]'
        self.eos_token = '[EOS]'

    def __len__(self):
        return 258

    def __call__(self,*args,**kargs):
        return self.encode(*args,**kargs)
    
    def encode(self,list_of_strings, pad_token_id=0,return_tensors=None,max_length=None,*args,**kargs):
        if type(list_of_strings) == str:
            list_of_strings = [list_of_strings]
        max_length = max([len(string) for string in list_of_strings])

        # create emtpy tensors
        # attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
        input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

        for idx, string in enumerate(list_of_strings):
            # make sure string is in byte format
            if not isinstance(string, bytes):
                string = str.encode(string)

            input_ids[idx, :len(string)] = torch.tensor([x + 2 for x in string])
            # attention_masks[idx, :len(string)] = 1
        if return_tensors == 'pt':
            return {'input_ids':input_ids}
        else:
            input_ids = input_ids.squeeze(0).tolist()
            # attention_masks = attention_masks.squeeze(0).tolist()
            return {'input_ids':input_ids}

    # Decoding
    def decode(self,outputs_ids):
        decoded_outputs = []
        for output_ids in outputs_ids.tolist():
            # transform id back to char IDs < 2 are simply transformed to ""
            decoded_outputs.append("".join([chr(x - 2) if x > 1 else "" for x in output_ids]))
        return decoded_outputs
