from torch.utils.data import Dataset
import json
import re

def data_filter(data_lines):
    answer_tag = ["A","B","C","D"]
    new_data_list = []
    for data_line in data_lines:
        data = json.loads(data_line)

        questions = data['questions'][:]
        article_spec_questions = data['specific_questions'][:]
        cloze_questions = data['cloze_questions'][:]
        general_questions = data['general_questions'][:]
        
        data['select_questions'] = article_spec_questions + cloze_questions + general_questions
        if (len(data['select_questions'])==0):
            continue
        
        # combine question and answer 
        for i,question in enumerate(data['select_questions']):
            q_id = questions.index(question)
            a_id = answer_tag.index(data['answers'][q_id])
            answer_text = data['options'][q_id][a_id]
            question = re.sub(r"www\..*\.com","",question)
            label_format = f"Q:{question} A:{answer_text}"

            # override select_questions
            data['select_questions'][i] = label_format
            # print(label_format)

        new_data_list.append(data)

    return new_data_list
