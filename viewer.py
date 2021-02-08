import argparse
from getkey import getkey, keys
import json
from torch.utils.data import Dataset,ConcatDataset
import os
import numpy as np
import time

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

def print_global_info(predict_dataset):
    # print col name
    print (u"{}[2J{}[;H".format(chr(27), chr(27)))
    question_scores = predict_dataset[0]['question_scores']
    
    scores = {}
    for question_score in question_scores:
        cos_s = ''
        for s_key in question_score.keys():
            scores[s_key] = 0.0 # init dict
            cos_s += "{:<10}".format(s_key[:8])
    print(cos_s)

    total_question_count = 0
    for i in range(len(predict_dataset)):
        data = predict_dataset[i]
        question_scores = data['question_scores']
        for question_score in question_scores:
            total_question_count+=1
            for s_key in question_score.keys():
                scores[s_key] += float(question_score[s_key])
    
    format_value = ''
    for s_key in scores.keys():
        f_s = "{:<10}".format(format_float(round(scores[s_key]/total_question_count*100,6)))
        format_value+=f_s
    print(format_value,end='\n\n')

    print("total test article:",len(predict_dataset))
    print("total question:",total_question_count)

    print(end='\n\n')
    print("exit:(q) or (o)")

if __name__ == "__main__":

    # config
    args = get_args()
    context = ''
    max_display = 600
    shift_step = 100
    current_index = 0

    # predict
    predict_dataset = PredictDataset(args.log_file)

    print_global_info(predict_dataset)

    #
    _shift_count = 0
    while True:
        context = predict_dataset[current_index]['article']
        
        print (u"{}[2J{}[;H".format(chr(27), chr(27)))
        print("index:",current_index,"context:",len(context),"shift:",_shift_count,"(h)elp",end='\n\n')
        
        # Predicts
        predict_questions_and_scores = []
        qs = predict_dataset[current_index]['questions']
        question_scores = predict_dataset[current_index]['question_scores']

        # col name
        for s in question_scores:
            cos_s = ''
            for s_key in s.keys():
                cos_s += "{:<10}".format(s_key[:8])

        # Predicts
        print("{:<62}".format('Predicts:')+cos_s)

        for q,s in zip(qs,question_scores):
            f_q = "{:<60}".format(q[:60])
            f_s = ""
            for s_key in s.keys():
                f_s += "{:<10}".format(format_float(round(float(s[s_key])*100,5)))
            predict_questions_and_scores.append(f_q+f_s)

        print('- '+'\n- '.join(predict_questions_and_scores),end='\n\n')

        # References
        print("References:")
        print('- '+'\n- '.join(predict_dataset[current_index]['labels']),end='\n\n')

        # Article
        print("Article:")
        print(context[0+_shift_count:max_display+_shift_count],end='\n\n')

        key = getkey() 
        
        if key == 'w':
            _shift_count -= shift_step
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
        elif key == 'i':
            print (u"{}[2J{}[;H".format(chr(27), chr(27)))
            current_index = int(input("index to go :"))
        elif key == 'h':
            print (u"{}[2J{}[;H".format(chr(27), chr(27)))
            print("up:(w) ,down:(s), next:(a), prev:(d), goto:(i), overview:(o), exit:(q)")
            time.sleep(3)
        elif key == 'o':
            print_global_info(predict_dataset)
            while True:
                if getkey() in ['q','o']:
                    break

        if _shift_count <0:_shift_count=0
        if current_index <0: current_index=len(predict_dataset)-1
        if current_index >len(predict_dataset)-1: current_index=0

