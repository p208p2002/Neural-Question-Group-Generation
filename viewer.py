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

if __name__ == "__main__":

    # config
    args = get_args()
    context = ''
    max_display = 600
    shift_step = 50
    current_index = 0

    # predict
    predict_dataset = PredictDataset(args.log_file)

    #    
    _shift_count = 0
    while True:
        context = predict_dataset[current_index]['article']
        
        print (u"{}[2J{}[;H".format(chr(27), chr(27)))
        print("index:",current_index,"context:",len(context),"shift:",_shift_count,"(h)elp",end='\n\n')
        
        # Predicts
        predict_questions_and_scores = []
        qs = predict_dataset[current_index]['questions']
        ss = predict_dataset[current_index]['question_scores']

        # col name
        for s in ss:
            cos_s = ''
            for s_key in s.keys():
                cos_s += "{:<10}".format(s_key[:8])

        # Predicts
        print("{:<62}".format('Predicts:')+cos_s)

        for q,s in zip(qs,ss):
            f_q = "{:<60}".format(q[:60])
            f_s = ""
            for s_key in s.keys():
                f_s += "{:<10}".format(format_float(round(float(s[s_key]),5)))
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
            print("up:(w) ,down:(s), next:(a), prev:(d), goto:(i), exit:(q)")
            time.sleep(3)

        if _shift_count <0:_shift_count=0
        if current_index <0: current_index=len(predict_dataset)-1
        if current_index >len(predict_dataset)-1: current_index=0

