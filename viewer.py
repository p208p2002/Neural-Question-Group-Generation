import argparse
from getkey import getkey, keys
import json
from torch.utils.data import Dataset,ConcatDataset
import os
import numpy as np
import time
import random

def format_float(num):
    return np.format_float_positional(num, trim='-')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file',type=str)
    args = parser.parse_args()
    args.batch_size = 1
    return args

class PredictDataset(Dataset):
    def __init__(self,f_path):
        with open(f_path,'r',encoding='utf-8') as f:
            self.data_lines = f.readlines()

    def __getitem__(self,index):
        return json.loads(self.data_lines[index])

    def __len__(self):
        return len(self.data_lines)

def print_global_info(predict_dataset,use_like_score):
    # print col name
    print (u"{}[2J{}[;H".format(chr(27), chr(27)))

    # search datsets
    dataset_names = set()
    dataset_names.add('all')
    gen_question_count = {}
    label_question_count = {}
    for i in range(len(predict_dataset)):
        data = predict_dataset[i]
        dataset_names.add(data['dataset_name'])
    
    for dataset_name in dataset_names:
        gen_question_count[dataset_name] = 0
        label_question_count[dataset_name] = 0

    # search col name
    _qi = 0
    col_names = []
    while True:
        try:
            if use_like_score:
                question_scores = predict_dataset[_qi]['question_scores']
            else:
                question_scores = predict_dataset[_qi]['unlike_question_scores']
            foramt_col_names = ''
            for score_key in question_scores[0].keys():
                if use_like_score:
                    foramt_col_names += "{:<20}".format(score_key[:18])
                else:
                    foramt_col_names += "{:<20}".format("*"+score_key[:17])
                if score_key not in col_names: col_names.append(score_key)
            break
        except:
            _qi+=1
            assert _qi < len(predict_dataset)-1,'search col name fail'
    
    # init score
    scores = {}
    classmate_scores = {}
    for dataset_name in dataset_names:
        scores[dataset_name] = {}
        classmate_scores[dataset_name] = {}
        for col_name in col_names:
            scores[dataset_name][col_name] = 0.0
            classmate_scores[dataset_name][col_name] = 0.0
            
    # total_question_count = 0
    for i in range(len(predict_dataset)):
        data = predict_dataset[i]
        if use_like_score:
            question_scores = data['question_scores']
        else:
            question_scores = data['unlike_question_scores']
        dataset_name = data['dataset_name']

        #
        for question_score in question_scores:
            # total_question_count+=1
            gen_question_count[dataset_name] += 1
            gen_question_count['all'] += 1
            for score_key in question_score.keys():
                scores[dataset_name][score_key] += float(question_score[score_key])
                scores['all'][score_key] += float(question_score[score_key])
        
        label_scores = data['unlike_label_scores']
        for label_score in label_scores:
            label_question_count[dataset_name] += 1
            label_question_count['all'] += 1
            for score_key in label_score.keys():
                classmate_scores[dataset_name][score_key] += float(label_score[score_key])
                classmate_scores['all'][score_key] += float(label_score[score_key])
    if use_like_score:
        print('-'*10,'predict references smilarity','-'*10)
    else:
        print('-'*10,'predict classmate smilarity','-'*10)
    for dataset_name in dataset_names:
        print(dataset_name)
        print(foramt_col_names)
        format_value = ''
        for score_key in scores[dataset_name].keys():
            f_s = "{:<20}".format(format_float(round((scores[dataset_name][score_key]/gen_question_count[dataset_name])*100,6)))
            format_value+=f_s
        print(format_value,end='\n\n')

    if not use_like_score:
        print('-'*10,'label classmate smilarity','-'*10)
        for dataset_name in dataset_names:
            print(dataset_name)
            print(foramt_col_names)
            format_value = ''
            for score_key in classmate_scores[dataset_name].keys():
                f_s = "{:<20}".format(format_float(round((classmate_scores[dataset_name][score_key]/label_question_count[dataset_name])*100,6)))
                format_value+=f_s
            print(format_value,end='\n\n')

    print("datasets:",dataset_names)
    print("generate questions:",gen_question_count)
    print("label questions:",label_question_count)
    print()

    total_question_coverage_score = 0
    total_label_coverage_score = 0
    for i in range(len(predict_dataset)):
        total_question_coverage_score += predict_dataset[i]['question_coverage_score']
        total_label_coverage_score += predict_dataset[i]['label_coverage_score']
    
    print('avg_question_coverage_score',round(total_question_coverage_score/gen_question_count['all']*100,5))
    print('avg_label_coverage_score',round(total_label_coverage_score/label_question_count['all']*100,5))
        

    print(end='\n\n')
    print("exit:(q) or (o)")

if __name__ == "__main__":

    # config
    args = get_args()
    context = ''
    max_display = 2000
    shift_step = 100
    current_index = 0
    max_question_len = 100

    # predict
    predict_dataset = PredictDataset(args.log_file)

    # check is m_race or not
    is_m_race = False
    if predict_dataset[0].get('levels',None) is not None:
        is_m_race = True
    
    # use like score or unlike score
    use_like_score = True

    #
    _shift_count = 0
    while True:
        context = predict_dataset[current_index]['article']
        dataset_name = predict_dataset[current_index]['dataset_name']
        
        print (u"{}[2J{}[;H".format(chr(27), chr(27)))
        print("index:",current_index,"context:",len(context),"shift:",_shift_count,"(h)elp")
        print('dataset_name:',dataset_name,'(u)se_like_score:',use_like_score,end='\n\n')
        print('question_coverage_score:',round(predict_dataset[current_index]['question_coverage_score']*100,5))
        print('label_coverage_score:',round(predict_dataset[current_index]['label_coverage_score']*100,5),end='\n\n')

        # Predicts
        predict_questions_and_scores = []
        qs = predict_dataset[current_index]['questions']

        if use_like_score:
            question_scores = predict_dataset[current_index]['question_scores']
        else:
            question_scores = predict_dataset[current_index]['unlike_question_scores']

        # col name
        cos_s = ''
        if len(question_scores) > 0:
            for score_key in question_scores[0].keys():
                if use_like_score:
                    cos_s += "{:<10}".format(score_key[:8])
                else:
                    score_key = "*"+score_key
                    cos_s += "{:<10}".format(score_key[:8])

        # Predicts
        print(("{:<%d}"%(max_question_len+4)).format('Predicts:')+cos_s)

        if is_m_race:
            levels = predict_dataset[current_index]['levels']
            for q,s,l in zip(qs,question_scores,levels):
                f_q = ("{:<%d}"%(max_question_len+2)).format(((str(l) +" "+ q)[:max_question_len]).replace("\n",""))
                f_s = ""
                for score_key in s.keys():
                    f_s += "{:<10}".format(format_float(round(float(s[score_key])*100,5)))
                predict_questions_and_scores.append(f_q+f_s)        
        else:
            for q,s in zip(qs,question_scores):
                f_q = ("{:<%d}"%(max_question_len+2)).format(q[:max_question_len].replace("\n",""))
                f_s = ""
                for score_key in s.keys():
                    f_s += "{:<10}".format(format_float(round(float(s[score_key])*100,5)))
                predict_questions_and_scores.append(f_q+f_s)

        print('- '+'\n- '.join(predict_questions_and_scores),end='\n\n')

        # References
        if use_like_score:
            print("References:")
            print('- '+'\n- '.join(predict_dataset[current_index]['labels']),end='\n\n')
        else:
            print(("{:<%d}"%(max_question_len+4)).format('References:')+cos_s)
            predict_questions_and_scores=[]
            question_scores = predict_dataset[current_index]['unlike_label_scores']
            for q,s in zip(predict_dataset[current_index]['labels'],question_scores):
                f_q = ("{:<%d}"%(max_question_len+2)).format(q[:max_question_len].replace("\n",""))
                f_s = ""
                for score_key in s.keys():
                    f_s += "{:<10}".format(format_float(round(float(s[score_key])*100,5)))
                predict_questions_and_scores.append(f_q+f_s)
            print('- '+'\n- '.join(predict_questions_and_scores),end='\n\n')
        

        # Article
        print("Article:")
        print(context[0+_shift_count:max_display+_shift_count],end='\n\n')

        key = getkey() 
        
        if key == 'w':
            _shift_count -= shift_step
        elif key == 'u':
            use_like_score = not use_like_score
        elif key == 's':
            _shift_count += shift_step
        elif key == 'a':
            _shift_count = 0
            current_index -= 1
        elif key == 'd':
            _shift_count = 0
            current_index += 1
        elif key == 'q':
            exit()
        elif key == 'r':
            current_index = random.randint(0,len(predict_dataset)-1)
        elif key == 'i':
            print (u"{}[2J{}[;H".format(chr(27), chr(27)))
            current_index = int(input("index to go :"))
        elif key == 'h':
            print (u"{}[2J{}[;H".format(chr(27), chr(27)))
            print("up:(w) ,down:(s), next:(a), prev:(d), goto:(i), random:(r), overview:(o), exit:(q)")
            time.sleep(3)
        elif key == 'o':
            print_global_info(predict_dataset,use_like_score)
            while True:
                key = getkey()
                if key in ['q','o']:
                    break
                elif key == 'u':
                    use_like_score = not use_like_score
                    print_global_info(predict_dataset,use_like_score)
                

        if _shift_count <0:_shift_count=0
        if current_index <0: current_index=len(predict_dataset)-1
        if current_index >len(predict_dataset)-1: current_index=0

