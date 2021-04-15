from torch.utils.data import Dataset
import json
import re
from .tokenizer import QUESTION_PREFIX_TOKEN,ANSWER_PREFIX_TOKEN
from loguru import logger

def data_filter_and_reconstruct(data_lines):
    answer_tag = ["A","B","C","D"]
    new_data_list = []
    for data_line in data_lines:
        data = json.loads(data_line)

        questions = data['questions'][:]
        article_spec_questions = data['specific_questions'][:]
        cloze_questions = data['cloze_questions'][:]
        general_questions = data['general_questions'][:]
        
        # continue, if no questions
        data['select_questions'] = article_spec_questions + cloze_questions + general_questions
        if (len(data['select_questions'])==0):
            continue
        
        # combine question and answer with format `[A:]answer[Q:]question`
        for i,question in enumerate(data['select_questions']):
            q_id = questions.index(question)
            a_id = answer_tag.index(data['answers'][q_id])
            answer_text = data['options'][q_id][a_id]
            question = re.sub(r"www\..*\.com","",question)
            label_format = f"{ANSWER_PREFIX_TOKEN}{answer_text}{QUESTION_PREFIX_TOKEN}{question}"

            # override select_questions
            data['select_questions'][i] = label_format

        new_data_list.append(data)

    return new_data_list

def separate_answer_and_question(raw_text):
    """
                   g0     g1    g2   g3
    re.compile(r"(\[A:\])(.*)(\[Q:\])(.*)").search("[A:]123[Q:]456").groups()
    ('[A:]', '123', '[Q:]', '456')
    """

    # try match answer first
    search = re.compile(r"(\[A:\])(.*)(\[Q:\])(.*)").search(raw_text)
    if search and len(search.groups()) == 4:
        search_groups = search.groups()
        answer_text = search_groups[1]
        question_text = search_groups[3]
        return {'answer_text':answer_text,'question_text':question_text}
    
    # try match question first
    search = re.compile(r"(\[Q:\])(.*)(\[A:\])(.*)").search(raw_text)
    if search and len(search.groups()) == 4:
        search_groups = search.groups()
        answer_text = search_groups[3]
        question_text = search_groups[1]
        return {'answer_text':answer_text,'question_text':question_text}
    
    logger.warning(f"qa separate with `{raw_text}` fail, return empty string")
    return {'answer_text':'','question_text':''}
