from torch.utils.data import DataLoader,Dataset
import os
import json
from .tokenizer import get_tokenizer
import torch

class RaceDataset(Dataset):
    def __init__(self,split_set,level,dataset_dir='datasets/RACE'):
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

        #
        self.tokenizer = get_tokenizer()
        self.sep_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.no_question = "There is no question to ask?"
        self.max_length = 1024
        self.max_context_length = 824

        # print(self.pad_token_id)
    
    def _prepare_input(self,context,label):
        tokenizer = self.tokenizer
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
        pad_labels = [-100]*self.max_length

        model_input['input_ids'] = (model_input['input_ids'] + pad_ids)[:self.max_length] 
        model_input['attention_mask'] = (model_input['attention_mask'] + pad_mask)[:self.max_length] 
        model_input['labels'] = (model_input['labels'] + pad_labels)[:self.max_length] 

        # convert to tensor
        for key in model_input.keys():
            model_input[key] = torch.LongTensor(model_input[key])
        
        return model_input
            
    def __getitem__(self,index):
        with open(self.all_file_paths[index],'r',encoding='utf-8') as f:
            data = json.load(f)
            context = data['article']
            _questions = data['questions']
            questions = []
            for _q in _questions:
                if _q[-1] == '?': # keep only type is question
                    questions.append(_q)
            
            if len(questions) == 0:
                questions.append(self.no_question)
            label = self.sep_token + self.sep_token.join(questions) + self.sep_token

            model_input = self._prepare_input(context,label)
            
        return model_input['input_ids'],model_input['attention_mask'],model_input['labels']
        
    def __len__(self):
        return len(self.all_file_paths)
