import os
from torch.utils.data import Dataset
import json

class MergeRaceDataset(Dataset):
    def __init__(self,split_set,level,dataset_dir='datasets/merge-race',eval_input=False):
        self.file_path  = os.path.join(dataset_dir,split_set,level+'.jsonl')
        self.data_lines = open(self.file_path,'r',encoding='utf-8').readlines()

        # attr
        self.count_general_question = 0
        self.count_article_spec_question = 0
        self.total_words = 0
        self.max_words = 0
        self.min_words = 99999
        self.total_quection_len = 0

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
            # if len(general_questions) == 0: continue; # keep only g-type >0

            data['general_questions'] = general_questions
            self.all_general_questions+=general_questions
            new_datas.append(data)
        self.datas = new_datas

    def __getitem__(self,index):
        data = self.datas[index]
        context = data['article']

        general_questions = data['general_questions'][:]
        self.count_general_question += len(general_questions)

        article_spec_questions = data['article_spec_questions'][:]
        self.count_article_spec_question += len(article_spec_questions)

        count_words = len(context.split())
        # print(count_words)
        self.total_words += count_words

        for q in (general_questions+article_spec_questions):
            self.total_quection_len += len(q.split())

        if count_words > self.max_words: self.max_words = count_words
        if count_words < self.min_words: self.min_words = count_words
        
    def __len__(self):
        return len(self.datas)

def compute(name,datasets):
    count_article = 0
    count_article_spec_question = 0
    count_general_question = 0
    total_quection_len = 0
    total_words = 0
    max_words = []
    min_words = []



    for dataset in datasets:
        for _ in dataset:
            pass

        count_article += len(dataset)
        count_article_spec_question += dataset.count_article_spec_question
        count_general_question += dataset.count_general_question
        total_quection_len += dataset.total_quection_len
        total_words += dataset.total_words
        max_words.append(dataset.max_words)
        min_words.append(dataset.min_words)
    
    print(name)
    print('-'*100)
    print('count_article',count_article)
    print('count_article_spec_question',count_article_spec_question)
    print('count_general_question',count_general_question)
    print('question_len_avg',total_quection_len/(count_general_question+count_article_spec_question))
    print('count_questions_avg',(count_article_spec_question+count_general_question)/count_article)
    print('total_words',total_words)
    print('avg_words',total_words/count_article)
    print('max_words',max(max_words))
    print('min_words',min(min_words))
    print()

if __name__ == "__main__":
    datasets = [
        MergeRaceDataset('train','middle'),
        MergeRaceDataset('train','high'),
        MergeRaceDataset('dev','middle'),
        MergeRaceDataset('dev','high'),
        MergeRaceDataset('test','middle'),
        MergeRaceDataset('test','high')
    ]

    train = [
        MergeRaceDataset('train','middle'),
        MergeRaceDataset('train','high')
    ]

    dev = [
        MergeRaceDataset('dev','middle'),
        MergeRaceDataset('dev','high'),
    ]

    test = [
        MergeRaceDataset('test','middle'),
        MergeRaceDataset('test','high')
    ]


    compute('all',datasets)
    compute('train',train)
    compute('dev',dev)
    compute('test',test)

    
        
    
    # print('-'*100)
    # print('count_article',count_article)
    # print('count_article_spec_question',count_article_spec_question)
    # print('count_general_question',count_general_question)
    # print('total_words',dataset.total_words,'avg_words',dataset.total_words/len(dataset))
    # print('max_words',dataset.max_words,'min_words',dataset.min_words)

    
        