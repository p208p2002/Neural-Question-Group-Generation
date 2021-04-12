from torch.utils.data import Dataset
import json

def data_filter(data_lines):
    new_data_list = []
    for data_line in data_lines:
        data = json.loads(data_line)
        article_spec_questions = data['specific_questions'][:]
        cloze_questions = data['cloze_questions'][:]
        
        if len(article_spec_questions) == 0: 
            continue
        
        data['select_questions'] = article_spec_questions + cloze_questions
        new_data_list.append(data)
    return new_data_list
