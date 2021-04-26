from torch.utils.data import Dataset
import json
import re
import random
from loguru import logger
from .tokenizer import QUESTION_PREFIX_TOKEN,ANSWER_PREFIX_TOKEN
from .scorer import scorers_runner
from .qgg_optimizer import optims_runner
from .argparser import get_general_args

def data_filter_and_reconstruct(data_lines,g_args=get_general_args()):
    answer_tag = ["A","B","C","D"]
    new_data_list = []
    for data_line in data_lines:
        data = json.loads(data_line)

        questions = data['questions'][:]
        article_spec_questions = data['specific_questions'][:]
        cloze_questions = data['cloze_questions'][:]
        general_questions = data['general_questions'][:]
        
        data['select_questions'] = []
        for use_subset in g_args.use_subsets:
            if use_subset == 'c-type':
                data['select_questions'] += cloze_questions
            elif use_subset == 'g-type':
                data['select_questions'] += general_questions
            elif use_subset == 's-type':
                data['select_questions'] += article_spec_questions
            else:
                raise Exception('condititon no match')
        
        # continue, if no questions
        if (len(data['select_questions'])==0):
            continue
        
        if g_args.gen_target == 'q-and-a':
            # combine question and answer with format `[A:]answer[Q:]question`
            for i,question in enumerate(data['select_questions']):
                q_id = questions.index(question)
                a_id = answer_tag.index(data['answers'][q_id])
                answer_text = data['options'][q_id][a_id]
                label_format = f"{ANSWER_PREFIX_TOKEN}{answer_text}{QUESTION_PREFIX_TOKEN}{question}"
                # override select_questions
                data['select_questions'][i] = label_format
                
        elif g_args.gen_target == 'only-q':
            # format to `[Q:]question`
            for i,question in enumerate(data['select_questions']):
                label_format = f"{QUESTION_PREFIX_TOKEN}{question}"
                # override select_questions
                data['select_questions'][i] = label_format

        else:
            raise Exception('`g_args.gen_target` no match')
        
        # clean noise for quesitons
        for i,question in enumerate(data['select_questions']):
            question = re.sub(r"www\..*\.com","",question)
            data['select_questions'][i] = question

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
    
    # try match only question
    search = re.compile(r"(\[Q:\])(.*)").search(raw_text)
    if search and len(search.groups()) == 2:
        search_groups = search.groups()
        question_text = search_groups[1]
        answer_text = ''
        return {'answer_text':answer_text,'question_text':question_text}
    
    logger.warning(f"qa separate with `{raw_text}` fail, return empty string")
    return {'answer_text':'','question_text':''}

def process_decode_questions(article,label_questions,decode_questions,args,qgg_optimizers,scorers,predict_logger,g_args = get_general_args()):
    """
    this func process the quesiotns that model generate
    we need to do some processing to group question
    and also log and eval
    """

    logger.debug(decode_questions)

    # clean qa pair format
    # the order of training target is `answer` -> `question`
    # but we changed to `question` -> `answer` here for readability
    decode_questions = [separate_answer_and_question(qa) for qa in decode_questions]
    
    # decode_questions may broken(e.g. not a qa pair)
    # try to fix it with repeat self
    _decode_questions = []
    for qa in decode_questions:
        if g_args.gen_target == 'q-and-a':
            if qa['question_text'] != "" and qa['answer_text'] !="":
                _decode_questions.append(qa)
        elif g_args.gen_target == 'only-q':
            if qa['question_text'] != "" and qa['answer_text'] =="":
                _decode_questions.append(qa)
        else:
            raise Exception('`g_args.gen_target` no match')
    
    if len(_decode_questions) < args.gen_n:
        logger.warning("some question is broken, `len(_decode_questions) < args.gen_n`, will try repeat self to filling")
    while len(_decode_questions) < args.gen_n:
        _decode_questions.append(_decode_questions[random.randint(0,len(_decode_questions)-1)])
    decode_questions = _decode_questions

    #
    if g_args.gen_target == 'q-and-a':
        decode_answers = [f"{qa['answer_text']}" for qa in decode_questions]
        decode_answers_ans_questions = [f"{qa['question_text']} {qa['answer_text']}" for qa in decode_questions]
    elif g_args.gen_target == 'only-q':
        decode_answers_ans_questions = [f"{qa['question_text']}" for qa in decode_questions]
    else:
        raise Exception('`g_args.gen_target` no match')

    label_questions = [separate_answer_and_question(qa) for qa in label_questions]
    label_questions = [f"{qa['question_text']}" for qa in label_questions]

    optims_results = optims_runner(
        optims=qgg_optimizers,
        optim_names=args.qgg_optims,
        condicate_questions=decode_answers_ans_questions,
        context=article
    )
    
    # filter out optims_results's qa to only q
    if g_args.gen_target == 'q-and-a':
        for decode_answer in decode_answers:
            for i,optims_result in enumerate(optims_results):
                optims_result = list(map(lambda qa: re.sub(re.escape(decode_answer)+r"$","",qa).strip(),optims_result))
                optims_results[i] = optims_result

    scorers_runner(
        scoers=scorers,
        optim_names=args.qgg_optims,
        optims_results=optims_results,
        label_questions=label_questions,
        article=article,
        predict_logger = predict_logger
    )

    return optims_results