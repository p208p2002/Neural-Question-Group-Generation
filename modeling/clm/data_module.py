from torch.utils.data import DataLoader,Dataset,ConcatDataset
import os
import json
from .tokenizer import get_tokenizer,RACE_BOS,_GENERAL_LEVEL,_MIDDLE_LEVEL
from .argparser import get_args
import torch
import pytorch_lightning as pl
import re
from .config import *
import random

class DataModule(pl.LightningDataModule):
    def __init__(self,args = get_args()):
        super().__init__()
        self.batch_size = args.batch_size
        self.args = args
    
    def get_dataset(self,stage,d_name):
         # set race dataset
        if d_name == 'race':
            train_dataset = ConcatDataset((RaceDataset('train','all'),RaceDataset('dev','all')))
            if stage == 'fit':
                test_dataset = RaceDataset('test','all',eval_input=False)
            elif stage == 'test':
                test_dataset = RaceDataset('test','all',eval_input=True)
        
        # set eqg dataset
        elif d_name == 'eqg':
            train_dataset = ConcatDataset((
                    EQGRaceDataset('train','middle'),
                    EQGRaceDataset('train','high'),
                    EQGRaceDataset('dev','middle'),
                    EQGRaceDataset('dev','high')
            ))
            if stage == 'fit':
                test_dataset = ConcatDataset((
                    EQGRaceDataset('test','middle',eval_input=False),
                    EQGRaceDataset('test','high',eval_input=False)
                ))
            elif stage == 'test':
                test_dataset = ConcatDataset((
                    EQGRaceDataset('test','middle',eval_input=True),
                    EQGRaceDataset('test','high',eval_input=True)
                ))
        
        # set general race
        elif d_name == 'g_race':
            train_dataset = ConcatDataset((
                    GeneralRaceDataset('train','middle'),
                    GeneralRaceDataset('train','high'),
                    GeneralRaceDataset('dev','middle'),
                    GeneralRaceDataset('dev','high')
            ))
            if stage == 'fit':
                test_dataset = ConcatDataset((
                    GeneralRaceDataset('test','middle',eval_input=False),
                    GeneralRaceDataset('test','high',eval_input=False)
                ))
            elif stage == 'test':
                test_dataset = ConcatDataset((
                    GeneralRaceDataset('test','middle',eval_input=True),
                    GeneralRaceDataset('test','high',eval_input=True)
                ))
        
        elif d_name == 'm_race':
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
    def set_config(self,dataset_name, eval_input, bos_token, max_length=MAX_LENGTH ,max_context_length=MAX_CONTEXT_LENGTH ):
        # general config
        self.tokenizer = get_tokenizer()
        self.sep_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id

        self.max_length = max_length
        self.max_context_length = max_context_length

        # spec config
        self.dataset_name = dataset_name
        self.eval_input = eval_input  
        self.bos_token = bos_token
        
    def prepare_input(self,context,label=None):
        tokenizer = self.tokenizer

        if label is None:
            model_input = tokenizer(context,return_tensors='pt',max_length=self.max_context_length,truncation=True)
            model_input['input_ids'] = model_input['input_ids'].squeeze(0) # fix shape bug with tokenizer
            model_input['attention_mask'] = model_input['attention_mask'].squeeze(0) # fix shape bug with tokenizer
            return model_input

        context_input = tokenizer(context)
        
        label_input = tokenizer(label)
        label_input['attention_mask'] = [0]*len(label_input['input_ids'])

        # limit context length
        context_input['input_ids'] = context_input['input_ids'][:self.max_length - len(label_input['input_ids'])]
        context_input['attention_mask'] = context_input['attention_mask'][:self.max_length - len(label_input['input_ids'])]

        model_input = {}
        model_input['input_ids'] = context_input['input_ids'] + label_input['input_ids']
        model_input['attention_mask'] = context_input['attention_mask'] + label_input['attention_mask']
        model_input['labels'] = model_input['input_ids'][:]
        for i,(l,a) in enumerate(zip(model_input['labels'],model_input['attention_mask'])):
            if a == 1: model_input['labels'][i] = -100

        # pad or limit to max length
        pad_ids = [self.pad_token_id]*self.max_length
        pad_mask = [0]*self.max_length
        pad_labels = [-100]*self.max_length

        model_input['input_ids'] = (model_input['input_ids'] + pad_ids)[:self.max_length] 
        model_input['attention_mask'] = (model_input['attention_mask'] + pad_mask)[:self.max_length] 
        model_input['labels'] = (model_input['labels'] + pad_labels)[:self.max_length]

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

class RaceDataset(Dataset,UtilsMixin):
    def __init__(self,split_set,level,dataset_dir='datasets/RACE',eval_input=False):
        super().__init__()
        assert split_set in ['dev','test','train']
        assert level in ['all','middle','high']
        self.all_file_paths = []
        for root, dirs, files in os.walk(os.path.join(dataset_dir,split_set)):
            for f in files:
                if level == 'all':
                    self.all_file_paths.append(os.path.join(root,f))
                elif root == os.path.join(dataset_dir,split_set,level):
                    self.all_file_paths.append(os.path.join(root,f))

        #
        self.set_config(dataset_name='race',eval_input=eval_input, bos_token=RACE_BOS)
            
    def __getitem__(self,index):
        with open(self.all_file_paths[index],'r',encoding='utf-8') as f:
            data = json.load(f)
            context = data['article']
            _questions = data['questions'][:]
            questions = []
            for _q in _questions:
                if _q[-1] == '?' and re.search('_',_q) is None: # keep only type is question
                    questions.append(_q)
            
            questions.append(self.tokenizer.eos_token)
            label = self.sep_token.join(questions) 

            if not self.eval_input:
                model_input = self.prepare_input(context + self.bos_token, label= label)
                return model_input['input_ids'],model_input['attention_mask'],model_input['labels']
            else:
                model_input = self.prepare_input(context + self.bos_token, label= None)
                return self.construct_eval_output(
                    self.dataset_name,
                    model_input['input_ids'],
                    model_input['attention_mask'],
                    questions[:-1],
                    data['article']
                )
            
    def __len__(self):
        return len(self.all_file_paths)

class EQGRaceDataset(Dataset,UtilsMixin):
    def __init__(self,split_set,level,dataset_dir='datasets/merge-race',eval_input=False):
        self.file_path  = os.path.join(dataset_dir,split_set,level+'.jsonl')
        self.data_lines = open(self.file_path,'r',encoding='utf-8').readlines()

        # config
        self.set_config(dataset_name='eqg',eval_input=eval_input,bos_token=RACE_BOS)

        # filter no question
        new_data = []
        for index in range(len(self.data_lines)):
            data = json.loads(self.data_lines[index])
            questions = data['article_spec_questions']
            if len(questions) > 0:
                new_data.append(self.data_lines[index])
        self.data_lines = new_data

    def __getitem__(self,index):
        data = json.loads(self.data_lines[index])
        context = data['article']
        questions = data['article_spec_questions'][:]
        questions.append(self.tokenizer.eos_token)
        label = self.sep_token.join(questions) 

        if not self.eval_input:
            model_input = self.prepare_input(context + self.bos_token, label= label)
            return model_input['input_ids'],model_input['attention_mask'],model_input['labels']
        else:
            model_input = self.prepare_input(context + self.bos_token, label= None)
            return self.construct_eval_output(
                self.dataset_name,
                model_input['input_ids'],
                model_input['attention_mask'],
                data['article_spec_questions'],
                data['article']
            )
    
    def __len__(self):
        return len(self.data_lines)

class GeneralRaceDataset(Dataset,UtilsMixin):
    def __init__(self,split_set,level,dataset_dir='datasets/merge-race',eval_input=False):
        self.file_path  = os.path.join(dataset_dir,split_set,level+'.jsonl')
        self.data_lines = open(self.file_path,'r',encoding='utf-8').readlines()

        # config
        self.set_config(dataset_name='g_race',eval_input=eval_input,bos_token=_GENERAL_LEVEL)        

        # keep only general question
        new_datas = []
        for data_line in self.data_lines:
            data = json.loads(data_line)
            article_spec_questions = data['article_spec_questions'][:]
            all_questions = data['questions'][:]

            general_questions = []
            for all_question in all_questions:
                if all_question not in article_spec_questions:
                    general_questions.append(all_question)
            if len(general_questions) >0: # remove no question
                data['general_questions'] = general_questions
                new_datas.append(data)
        self.datas = new_datas

    def __getitem__(self,index):
        data = self.datas[index]
        context = data['article']
        general_questions = data['general_questions'][:]

        #
        general_questions.append(self.tokenizer.eos_token)
        label = self.sep_token.join(general_questions) 

        if not self.eval_input:
            model_input = self.prepare_input(context + self.bos_token, label= label)
            return model_input['input_ids'],model_input['attention_mask'],model_input['labels']
        else:
            model_input = self.prepare_input(context + self.bos_token, label= None)
            return self.construct_eval_output(
                self.dataset_name,
                model_input['input_ids'],
                model_input['attention_mask'],
                data['general_questions'],
                data['article']
            )
    
    def __len__(self):
        return len(self.datas)

class MergeRaceDataset(Dataset,UtilsMixin):
    def __init__(self,split_set,level,dataset_dir='datasets/merge-race',eval_input=False):
        self.file_path  = os.path.join(dataset_dir,split_set,level+'.jsonl')
        self.data_lines = open(self.file_path,'r',encoding='utf-8').readlines()

        # config
        self.set_config(dataset_name='m_race',eval_input=eval_input,bos_token=None)
        self.bos_tokens = [_GENERAL_LEVEL+" ",_MIDDLE_LEVEL+" "]

        # attr
        self.count_general_question = 0
        self.count_article_spec_question = 0

        # select general question
        self.all_general_questions = []
        new_datas = []
        for data_line in self.data_lines:
            data = json.loads(data_line)
            article_spec_questions = data['article_spec_questions'][:]
            if len(article_spec_questions) == 0: continue; # keep only s-type >0
            all_questions = data['questions'][:]
            # if len(all_questions) == 0: continue
            
            general_questions = []
            for all_question in all_questions:
                if all_question not in article_spec_questions:
                    general_questions.append(all_question)   
            if len(general_questions) == 0: continue; # keep only g-type >0

            data['general_questions'] = general_questions
            self.all_general_questions+=general_questions
            new_datas.append(data)
        self.datas = new_datas
    
    def random_general_question(self):
        return self.all_general_questions[random.randint(0,len(self.all_general_questions)-1)]

    def __getitem__(self,index):
        data = self.datas[index]
        context = data['article']

        general_questions = data['general_questions'][:]
        self.count_general_question += len(general_questions)
        general_questions = [self.bos_tokens[0]+ q for q in general_questions]
        random.shuffle(general_questions)

        article_spec_questions = data['article_spec_questions'][:]
        self.count_article_spec_question += len(article_spec_questions)
        article_spec_questions = [self.bos_tokens[1]+ q for q in article_spec_questions]
        random.shuffle(article_spec_questions)

        all_questions_with_bos = general_questions[:1] + article_spec_questions[:1]
        # random.shuffle(all_questions_with_bos)
        all_questions_with_bos.append(self.tokenizer.eos_token)
        
        label = ' '.join(all_questions_with_bos)

        # print('s_type:',len(article_spec_questions),'g_tpye:',len(general_questions),self.random_general_question())

        if not self.eval_input:
            # context_shift = random.randint(0,200)
            model_input = self.prepare_input(context + self.tokenizer.sep_token, label= label)
            return model_input['input_ids'],model_input['attention_mask'],model_input['labels']
        else:
            model_input = self.prepare_input(context + self.tokenizer.sep_token, label= None)
            return self.construct_eval_output(
                self.dataset_name,
                model_input['input_ids'],
                model_input['attention_mask'],
                data['general_questions'] + data['article_spec_questions'],
                data['article']
            )
    
    def __len__(self):
        return len(self.datas)
