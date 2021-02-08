import argparse
from getkey import getkey, keys
import json
from torch.utils.data import Dataset,ConcatDataset
import os

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
    max_display = 1500
    shift_step = 100
    current_index = 0

    # predict
    predict_dataset = PredictDataset(args.log_file)

    #    
    _shift_count = 0
    while True:
        context = predict_dataset[current_index]['article']\
        + '\n\n'\
        + 'targets:\n'\
        + '- '+'\n- '.join(predict_dataset[current_index]['labels'])\
        + '\n\n'\
        + 'predicts:\n'\
        + '- '+'\n- '.join(predict_dataset[current_index]['questions'])\
        # + 'predicts:\n'
        
        print (u"{}[2J{}[;H".format(chr(27), chr(27)))
        print(current_index,len(context),_shift_count)
        print(context[0+_shift_count:max_display+_shift_count])

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

        
        if _shift_count <0:_shift_count=0
        if current_index <0: current_index=len(predict_dataset)-1
        if current_index >len(predict_dataset)-1: current_index=0

