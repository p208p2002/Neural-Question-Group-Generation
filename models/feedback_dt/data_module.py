from torch.utils.data import DataLoader,Dataset,ConcatDataset
import os
import json
from .tokenizer import get_tokenizer,GENED_TOKEN,GENED_SEP
from .argparser import get_args
import torch
import pytorch_lightning as pl
import re
from .config import *
import random
from utils import make_stop_word_ids,ignore_pad_token_ids
from transformers.models.bart.modeling_bart import shift_tokens_right

#
import torch
import re
from torch._six import container_abcs, string_classes, int_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, container_abcs.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def default_collate(batch):
    
    # print(numel)
    count_bundle_len = len(batch[0])
    # new_batch = []
    # print(count_data)
    for i,bundle in enumerate(batch):
        for j,data in enumerate(bundle):
            print(i,j,data.shape)

        
        
    # r"""Puts each data field into a tensor with outer dimension batch size"""

    # # print("@@@@@@@@")
    # # print(batch[0][0].shape,len(batch[0]))
    # # exit()
    

    # elem = batch[0]
    # elem_type = type(elem)
    # if isinstance(elem, torch.Tensor):
    #     print("in")
    #     out = None
    #     if torch.utils.data.get_worker_info() is not None:
    #         # If we're in a background process, concatenate directly into a
    #         # shared memory tensor to avoid an extra copy
    #         numel = sum([x.numel() for x in batch])
    #         print(numel)
    #         storage = elem.storage()._new_shared(numel)
    #         out = elem.new(storage)
    #     return torch.stack(batch, 0, out=out)
    # elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
    #         and elem_type.__name__ != 'string_':
    #     if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
    #         # array of string classes and object
    #         if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
    #             raise TypeError(default_collate_err_msg_format.format(elem.dtype))

    #         return default_collate([torch.as_tensor(b) for b in batch])
    #     elif elem.shape == ():  # scalars
    #         return torch.as_tensor(batch)
    # elif isinstance(elem, float):
    #     return torch.tensor(batch, dtype=torch.float64)
    # elif isinstance(elem, int_classes):
    #     return torch.tensor(batch)
    # elif isinstance(elem, string_classes):
    #     return batch
    # elif isinstance(elem, container_abcs.Mapping):
    #     return {key: default_collate([d[key] for d in batch]) for key in elem}
    # elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
    #     return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    # elif isinstance(elem, container_abcs.Sequence):
    #     # check to make sure that the elements in batch have consistent size
    #     it = iter(batch)
    #     elem_size = len(next(it))
    #     if not all(len(elem) == elem_size for elem in it):
    #         raise RuntimeError('each element in list of batch should be of equal size')
    #     transposed = zip(*batch)
    #     return [default_collate(samples) for samples in transposed]

    # raise TypeError(default_collate_err_msg_format.format(elem_type))



class DataModule(pl.LightningDataModule):
    def __init__(self,args = get_args()):
        super().__init__()
        self.batch_size = args.batch_size
        self.args = args
    
    def get_dataset(self,stage,d_name):
        if d_name == 'm_race':
            train_dataset = ConcatDataset((
                    MergeRaceDataset('train','middle'),
                    MergeRaceDataset('train','high'),
                    MergeRaceDataset('dev','middle'),
                    MergeRaceDataset('dev','high')
            ))
            if stage == 'fit':
                test_dataset = ConcatDataset((
                    MergeRaceDataset('test','middle',eval_input=False),
                    MergeRaceDataset('test','high',eval_input=False)
                ))
            elif stage == 'test':
                test_dataset = ConcatDataset((
                    MergeRaceDataset('test','middle',eval_input=True),
                    MergeRaceDataset('test','high',eval_input=True)
                ))
        
        # match fail
        else:
            assert False,'no dataset match'
        
        return train_dataset,test_dataset
        
    def setup(self, stage=None):
        train_datasets,test_datasets = [],[]
        print('stage:',stage,', using datasets:',self.args.datasets)
        for d_name in self.args.datasets:
            print('loading `%s`...'%d_name,end='\r')
            train_dataset,test_dataset = self.get_dataset(stage=stage,d_name=d_name)
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
            print('loading `%s`...finish'%d_name)
        
        self.train_dataset = ConcatDataset(train_datasets)
        self.test_dataset = ConcatDataset(test_datasets)
       
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

class UtilsMixin():
    def set_config(self,dataset_name, eval_input, bos_token, max_length=MAX_LENGTH ):
        # general config
        self.tokenizer = get_tokenizer()
        self.sep_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = max_length

        self.stop_word_ids = make_stop_word_ids(self.tokenizer)

        # spec config
        self.dataset_name = dataset_name
        self.eval_input = eval_input  
        self.bos_token = bos_token
    
    def prepare_input(self,context,gend_labels=[],label=None,gened_token=GENED_TOKEN,g_sep=GENED_SEP,max_length=MAX_LENGTH):
        tokenizer = self.tokenizer
    
        model_input = {}
        
        pad_token_id = tokenizer.pad_token_id
        _gend_labels = gened_token + g_sep.join(gend_labels) + gened_token
        context_encodings = tokenizer(_gend_labels+context, padding='max_length', max_length=max_length, truncation=True, add_special_tokens=False)
        
        model_input['encoder_input_ids'] = context_encodings['input_ids']
        model_input['encoder_attention_mask'] = context_encodings['attention_mask']
        
        # pos decoder input
        decoder_input = tokenizer.bos_token + label
        decoder_encodings = tokenizer(decoder_input, padding='max_length', max_length=max_length, truncation=True, add_special_tokens=False)
        
        model_input['decoder_input_ids'] = decoder_encodings['input_ids']
        model_input['decoder_attention_mask'] = decoder_encodings['attention_mask']
        
        # pos decoder label
        decoder_label = label + tokenizer.eos_token
        decoder_label_encodings = tokenizer(
            decoder_label,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=False
        )
        model_input['decoder_labels'] = decoder_encodings['input_ids']
        model_input['decoder_labels'] = ignore_pad_token_ids(model_input['decoder_labels'],pad_token_id)
        
        # neg
        max_length = 128
        n_decoder_inputs = []
        n_decoder_attention_mask = []
        n_decoder_labels = []
        if len(gend_labels) >0:
            for gend_label in gend_labels:
                # neg decoder input
                n_decoder_input = tokenizer.bos_token + gend_label
                n_decoder_input_encodings = tokenizer(n_decoder_input, padding='max_length', max_length=max_length, truncation=True, add_special_tokens=False)
                n_decoder_inputs.append(n_decoder_input_encodings['input_ids'])
                n_decoder_attention_mask.append(n_decoder_input_encodings['attention_mask'])
                
                # neg decoder label
                n_decoder_label = gend_label + tokenizer.eos_token
                n_decoder_label_encodings = tokenizer(
                    n_decoder_label,
                    padding='max_length',
                    max_length=max_length,
                    truncation=True,
                    add_special_tokens=False,
                    return_attention_mask=False
                )
                n_decoder_labels.append(ignore_pad_token_ids(n_decoder_label_encodings['input_ids'],pad_token_id))
        while len(n_decoder_inputs) < 6:
            n_decoder_inputs.append([tokenizer.pad_token_id]*max_length)
            n_decoder_attention_mask.append([0]*max_length)
            n_decoder_labels.append([-100]*max_length)
        assert len(n_decoder_inputs) == 6
        

        #
        model_input['n_decoder_inputs'] = n_decoder_inputs
        model_input['n_decoder_attention_mask'] = n_decoder_attention_mask
        model_input['n_decoder_labels'] = n_decoder_labels
        
        return model_input
        
    def construct_eval_output(self,dataset_name,input_ids,attention_mask,label_questions,article):
        """
        dataset_name: str
        input_ids: tensor
        attention_mask: tensor
        label_questions: list[str]
        article: str
        """
        return dataset_name,input_ids,attention_mask,label_questions,article

class MergeRaceDataset(Dataset,UtilsMixin):
    def __init__(self,split_set,level,dataset_dir='datasets/EQG-RACE-PLUS',eval_input=False):
        self.file_path  = os.path.join(dataset_dir,split_set,level+'.jsonl')
        self.data_lines = open(self.file_path,'r',encoding='utf-8').readlines()

        # config
        self.set_config(dataset_name='m_race',eval_input=eval_input,bos_token=None)
        self.sep_token = self.tokenizer.sep_token

        # select general question
        self.all_general_questions = []
        new_datas = []
        for data_line in self.data_lines:
            data = json.loads(data_line)
            article_spec_questions = data['specific_questions'][:]
            cloze_questions = data['cloze_questions'][:]
            # if len(article_spec_questions) == 0 and len(cloze_questions) == 0: 
            #     continue
            if len(article_spec_questions) == 0: 
                continue
            
            new_datas.append(data)
        self.datas = new_datas


    def __getitem__(self,index):
        data = self.datas[index]
        context = data['article']
        article_spec_questions = data['specific_questions']
        cloze_questions = data['cloze_questions']

        all_questions = article_spec_questions + cloze_questions
        random.shuffle(all_questions)

        # 
        random_select_question_for_label = random.randint(0,len(all_questions)-1)        
        question_for_label = all_questions.pop(random_select_question_for_label)

        # we random select for state
        try:
            random_state_rage = random.randint(0,len(all_questions)-1)
        except:
            random_state_rage = 0
        
        all_questions = all_questions[:random_state_rage]

        #
        if not self.eval_input: # train
            gend_labels = all_questions[:]
            model_input = self.prepare_input(
                context=context,
                gend_labels=gend_labels,
                label=WARN_UP_TOKEN + question_for_label + self.tokenizer.eos_token,
                )
            
            for key,item in model_input.items():
                item = torch.LongTensor(item)
                model_input[key] = item
                # print(key,model_input[key].shape)
            
            return (
                # context
                model_input['encoder_input_ids'],
                model_input['encoder_attention_mask'],
                # possitive
                model_input['decoder_input_ids'],
                model_input['decoder_attention_mask'],
                model_input['decoder_labels'],
                # # negative
                model_input['n_decoder_inputs'],
                model_input['n_decoder_attention_mask'],
                model_input['n_decoder_labels']
            )
        else:
            assert False
            # model_input = self.prepare_input(context, label= None)
            # return self.construct_eval_output(
            #     self.dataset_name,
            #     model_input['input_ids'],
            #     model_input['attention_mask'],
            #     data['specific_questions']+data['cloze_questions'],
            #     context
            # )
    
    def __len__(self):
        return len(self.datas)