from torch.utils.data import DataLoader,Dataset,ConcatDataset
import os
import json
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
import pytorch_lightning as pl
import re
args = get_args()

class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = args.batch_size
        
    def setup(self, stage=None):
        if args.dataset == 'race':
            if stage == 'fit':
                self.train_dataset = ConcatDataset((RaceDataset('train','all'),RaceDataset('dev','all')))
                self.test_dataset = RaceDataset('test','all',no_label=False)
            elif stage == 'test':
                self.test_dataset = RaceDataset('test','all',no_label=True)
        elif args.dataset == 'eqg':
            if stage == 'fit':
                self.train_dataset = ConcatDataset((
                    EQGRaceDataset('train','middle'),
                    EQGRaceDataset('train','high'),
                    EQGRaceDataset('dev','middle'),
                    EQGRaceDataset('dev','high')
                    ))
                self.test_dataset = ConcatDataset((
                    EQGRaceDataset('test','middle',no_label=False),
                    EQGRaceDataset('test','high',no_label=False)
                    ))
            elif stage == 'test':
                self.test_dataset = ConcatDataset((
                    EQGRaceDataset('test','middle',no_label=True),
                    EQGRaceDataset('test','high',no_label=True)
                    ))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

class UtilsMixin():
    def _prepare_input(self,context,label=None):
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
        context_input['input_ids'] = context_input['input_ids'][:self.max_context_length]
        context_input['attention_mask'] = context_input['attention_mask'][:self.max_context_length]

        model_input = {}
        model_input['input_ids'] = context_input['input_ids'] + label_input['input_ids']
        model_input['attention_mask'] = context_input['attention_mask'] + label_input['attention_mask']
        model_input['labels'] = model_input['input_ids'][:]
        for i,(l,a) in enumerate(zip(model_input['labels'],model_input['attention_mask'])):
            if a == 1: model_input['labels'][i] = -100

        # pad or limit to max length
        pad_ids = [self.pad_token_id]*self.max_length
        pad_mask = [0]*self.max_length
        pad_labels = [self.pad_token_id]*self.max_length

        model_input['input_ids'] = (model_input['input_ids'] + pad_ids)[:self.max_length] 
        model_input['attention_mask'] = (model_input['attention_mask'] + pad_mask)[:self.max_length] 
        model_input['labels'] = (model_input['labels'] + pad_labels)[:self.max_length] 

        # convert to tensor
        for key in model_input.keys():
            model_input[key] = torch.LongTensor(model_input[key])
        
        return model_input

class RaceDataset(Dataset,UtilsMixin):
    def __init__(self,split_set,level,dataset_dir='datasets/RACE',no_label=False):
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
        print(split_set,level,dataset_dir,len(self))

        # config
        self.tokenizer = get_tokenizer()
        self.sep_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = 1024
        self.max_context_length = 850
        self.no_label = no_label    
            
    def __getitem__(self,index):
        with open(self.all_file_paths[index],'r',encoding='utf-8') as f:
            data = json.load(f)
            context = data['article']
            _questions = data['questions']
            questions = []
            for _q in _questions:
                if _q[-1] == '?' and re.search('_',_q) is None: # keep only type is question
                    questions.append(_q)
            
            questions.append(self.tokenizer.eos_token)
            label = self.sep_token.join(questions) 

            if not self.no_label:
                model_input = self._prepare_input(context + self.tokenizer.bos_token, label= label)
                return model_input['input_ids'],model_input['attention_mask'],model_input['labels']
            else:
                model_input = self._prepare_input(context + self.tokenizer.bos_token, label= None)
                return model_input['input_ids'],model_input['attention_mask']
            
    def __len__(self):
        return len(self.all_file_paths)

class EQGRaceDataset(Dataset,UtilsMixin):
    def __init__(self,split_set,level,dataset_dir='datasets/merge-race',no_label=False):
        self.file_path  = os.path.join(dataset_dir,split_set,level,level+'.jsonl')
        self.data_lines = open(self.file_path,'r',encoding='utf-8').readlines()

        # config
        self.tokenizer = get_tokenizer()
        self.sep_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = 1024
        self.max_context_length = 850
        self.no_label = no_label

    def __getitem__(self,index):
        data = json.loads(self.data_lines[index])
        context = data['article']
        questions = data['acticle_spec_questions']        
        questions.append(self.tokenizer.eos_token)
        label = self.sep_token.join(questions) 

        if not self.no_label:
            model_input = self._prepare_input(context + self.tokenizer.bos_token, label= label)
            return model_input['input_ids'],model_input['attention_mask'],model_input['labels']
        else:
            model_input = self._prepare_input(context + self.tokenizer.bos_token, label= None)
            return model_input['input_ids'],model_input['attention_mask']
    
    def __len__(self):
        return len(self.data_lines)
