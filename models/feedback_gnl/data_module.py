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
        stop_word_ids = self.stop_word_ids
        model_input = {}
        
        pad_token_id = tokenizer.pad_token_id
        _gend_labels = gened_token + g_sep.join(gend_labels) + gened_token
        context_encodings = tokenizer(_gend_labels+context, padding='max_length', max_length=max_length, truncation=True, add_special_tokens=False)
        
        model_input['encoder_input_ids'] = context_encodings['input_ids']
        model_input['encoder_attention_mask'] = context_encodings['attention_mask']
        
        # pos decoder input
        max_length = 32
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
        model_input['decoder_labels'] = decoder_label_encodings['input_ids']
        model_input['decoder_labels'] = ignore_pad_token_ids(model_input['decoder_labels'],pad_token_id)
        
        # neg
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
                n_decoder_label = gend_label + tokenizer.pad_token # usd pad_token for eos
                n_decoder_label_encodings = tokenizer(
                    n_decoder_label,
                    padding='max_length',
                    max_length=max_length,
                    truncation=True,
                    add_special_tokens=False,
                    return_attention_mask=False
                )
                n_decoder_label_input_ids = n_decoder_label_encodings['input_ids']
                # ignore stopwords
                for i,l_id in enumerate(n_decoder_label_input_ids):
                    if l_id in stop_word_ids:
                        n_decoder_label_input_ids[i] = -100
                
                # print(n_decoder_label_input_ids)
                n_decoder_labels.append(ignore_pad_token_ids(n_decoder_label_input_ids,pad_token_id))
        while len(n_decoder_inputs) < 6:
            n_decoder_inputs.append([tokenizer.pad_token_id]*max_length)
            n_decoder_attention_mask.append([0]*max_length)
            n_decoder_labels.append([-100]*max_length)
        assert len(n_decoder_inputs) == 6
        

        #
        model_input['n_decoder_inputs'] = n_decoder_inputs
        model_input['n_decoder_attention_mask'] = n_decoder_attention_mask
        model_input['n_decoder_labels'] = n_decoder_labels

        for key,item in model_input.items():
            item = torch.LongTensor(item)
            model_input[key] = item
        
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

        #
        # data['specific_questions'] = [q.replace("?"," ?") for q in data['specific_questions']]
        # data['cloze_questions'] =  [q.replace("?"," ?") for q in data['cloze_questions']]

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
                gend_labels= gend_labels,
                label= question_for_label,
                )
            
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
            model_input = self.prepare_input(context, label= '')
            return self.construct_eval_output(
                self.dataset_name,
                model_input['encoder_input_ids'],
                model_input['encoder_attention_mask'],
                data['specific_questions']+data['cloze_questions'],
                context
            )
    
    def __len__(self):
        return len(self.datas)
