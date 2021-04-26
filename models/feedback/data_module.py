from torch.utils.data import DataLoader,Dataset,ConcatDataset
import os
import json
from .tokenizer import get_tokenizer,GENED_TOKEN
from .argparser import get_args
import torch
import pytorch_lightning as pl
import re
from .config import *
import random
from transformers.models.bart.modeling_bart import shift_tokens_right
from utils.data_process import data_filter_and_reconstruct

class DataModule(pl.LightningDataModule):
    def __init__(self,args = get_args()):
        super().__init__()
        self.batch_size = args.batch_size
        self.args = args
        
    def setup(self, stage=None):
        self.train_dataset = ConcatDataset((
            MergeRaceDataset('train','middle'),
            MergeRaceDataset('train','high'),
            MergeRaceDataset('dev','middle'),
            MergeRaceDataset('dev','high')
        ))

        if stage == 'fit':
            self.test_dataset = ConcatDataset((
                MergeRaceDataset('test','middle',eval_input=False),
                MergeRaceDataset('test','high',eval_input=False)
            ))
        elif stage == 'test':
            self.test_dataset = ConcatDataset((
                MergeRaceDataset('test','middle',eval_input=True),
                MergeRaceDataset('test','high',eval_input=True)
            ))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=1, batch_size=1, shuffle=False)

class UtilsMixin():
    def set_config(self,dataset_name, eval_input, bos_token ):
        # general config
        self.tokenizer = get_tokenizer()
        self.sep_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id

        # spec config
        self.dataset_name = dataset_name
        self.eval_input = eval_input  
        self.bos_token = bos_token
        
    def prepare_input(self,context,label=None,is_negative=False):
        tokenizer = self.tokenizer
        # stop_word_ids = self.stop_word_ids
        pad_token_id = tokenizer.pad_token_id
        input_encodings = tokenizer(context, padding='max_length' if label is not None else False, max_length=MAX_LENGTH, truncation=True, add_special_tokens=False)
        
        if label is not None:
            decoder_input_ids = []
            labels = []
            target_encodings = tokenizer(label, padding='max_length', max_length=MAX_QUESTION_LENGTH, truncation=True, add_special_tokens=False)
            for target_encoding_id in target_encodings['input_ids']:
                decoder_input_ids.append(target_encoding_id)
                if target_encoding_id != pad_token_id:
                    labels.append(target_encoding_id)
                else:
                    labels.append(-100) # ignore "pad token" in label

            decoder_input_ids = [tokenizer.bos_token_id] + decoder_input_ids[:-1] # decoder_input_ids is `2` in BART

        else:
            labels = None
            decoder_input_ids = None
        
        #   
        model_input = {
            'input_ids':input_encodings['input_ids'],
            'attention_mask':input_encodings['attention_mask'],
            'decoder_input_ids': decoder_input_ids,
            'labels': labels
        }
        if label is None: 
            del model_input['labels']
            del model_input['decoder_input_ids']

        # convert to tensor
        for key in model_input.keys():
            model_input[key] = torch.LongTensor(model_input[key])

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

        # filter question group size < 0
        # and combine answer and question to `select_questions`
        self.datas = data_filter_and_reconstruct(self.data_lines)

    def __getitem__(self,index):
        data = self.datas[index]
        context = data['article']

        all_questions = data['select_questions'][:]
        random.shuffle(all_questions)

        # sample one question from question group as label
        random_select_question_for_label = random.randint(0,len(all_questions)-1)        
        question_for_label = all_questions.pop(random_select_question_for_label)

        # prune question group size for simulation generate state
        # t0: [gen][gen]context
        # t1: [gen]a1[gen]context
        # t2: [gen]a1,a2[gen]context
        if len(all_questions) >1:
            random_state_rage = random.randint(0,len(all_questions)-1)
        else:
            random_state_rage = 0
        all_questions = all_questions[:random_state_rage]

        #
        if not self.eval_input: # for training data
            context = GENED_TOKEN +  self.sep_token.join([re.sub(r"\[Q:\].*$","",qa)  for qa in all_questions]) + GENED_TOKEN + context
            label = question_for_label + self.tokenizer.eos_token
            model_input = self.prepare_input(context, label= label)

            # select the last question for negative label
            # t0: no negative label
            # t1: q1 for negative label
            # t2: q2 for negative label
            if len(all_questions)>0:
                negative_sample_label =  all_questions.pop(-1) + self.tokenizer.eos_token
                # the label contain `question` and `answer`, but we only need `answer` for negative loss
                negative_sample_label = re.sub(r"\[Q:\].*$","",negative_sample_label) 
                negative_model_input = self.prepare_input(context, label= negative_sample_label, is_negative = True)

                model_input['n_decoder_input_ids'] = negative_model_input['decoder_input_ids']
                model_input['n_labels'] = negative_model_input['labels']
                
                # no punish for `[Q:]` and `[A:]`
                model_input['n_labels'] = torch.where(
                    model_input['n_labels'] != self.tokenizer.question_prefix_token_id,
                    model_input['n_labels'],
                    -100
                )
                model_input['n_labels'] = torch.where(
                    model_input['n_labels'] != self.tokenizer.answer_prefix_token_id,
                    model_input['n_labels'],
                    -100
                )                
            else:
                model_input['n_decoder_input_ids'] = torch.LongTensor([self.tokenizer.pad_token_id]*len(model_input['labels']))
                model_input['n_labels'] = torch.LongTensor([-100]*len(model_input['labels']))
            
            return (
                # context
                model_input['input_ids'],
                model_input['attention_mask'],
                # possitive
                model_input['decoder_input_ids'],
                model_input['labels'],
                # negative
                model_input['n_decoder_input_ids'],
                model_input['n_labels'],
            )
        else:
            model_input = self.prepare_input(context, label= None)
            return self.construct_eval_output(
                self.dataset_name,
                model_input['input_ids'],
                model_input['attention_mask'],
                data['select_questions'],
                context
            )
    
    def __len__(self):
        return len(self.datas)
