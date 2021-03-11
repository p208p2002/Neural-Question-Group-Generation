from torch.utils.data import DataLoader,Dataset,ConcatDataset
import os
import json
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
import pytorch_lightning as pl
import re
from .config import *
import random
from utils import make_stop_word_ids

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
        
    def prepare_input(self,context,label=None,is_negative=False):
        tokenizer = self.tokenizer
        stop_word_ids = self.stop_word_ids
        pad_token_id = tokenizer.pad_token_id
        input_encodings = tokenizer(context, padding='max_length' if label is not None else False, max_length=self.max_length, truncation=True, add_special_tokens=False)
        
        if label is not None:
            labels = []
            target_encodings = tokenizer(label, padding='max_length', max_length=self.max_length, truncation=True, add_special_tokens=False)
            for target_encoding_id in target_encodings['input_ids']:
                if target_encoding_id != pad_token_id:
                    labels.append(target_encoding_id)
                else:
                    labels.append(-100)
            #
            if is_negative: # ignore head and eos (set to -100) 
                for i,l_id in enumerate(labels):
                    if l_id in stop_word_ids:
                        labels[i] = -100

        else:
            labels = None

        #   
        model_input = {
            'input_ids':input_encodings['input_ids'],
            'attention_mask':input_encodings['attention_mask'],
            'labels': labels
        }
        if label is None: del model_input['labels']

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
            context = self.sep_token + self.sep_token.join(all_questions) + context            
            label = "^%" + question_for_label + self.tokenizer.eos_token
            #
            model_input = self.prepare_input(context, label= label)

            # random select for negative
            if len(all_questions)>0:
                random.shuffle(all_questions)
                negative_sample_label = "^%" + all_questions.pop(0) + self.tokenizer.eos_token
                model_input['negative_sample_ids'] = self.prepare_input(context, label= negative_sample_label, is_negative = True)['labels']
            else:
                model_input['negative_sample_ids'] = torch.LongTensor([-100]*len(model_input['labels']))
            
            return model_input['input_ids'],model_input['attention_mask'],model_input['labels'],model_input['negative_sample_ids']
        else:
            model_input = self.prepare_input(context, label= None)
            return self.construct_eval_output(
                self.dataset_name,
                model_input['input_ids'],
                model_input['attention_mask'],
                data['specific_questions']+data['cloze_questions'],
                context
            )
    
    def __len__(self):
        return len(self.datas)
