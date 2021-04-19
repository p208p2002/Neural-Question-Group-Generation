import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
import re
import os
import json
from .config import *
from utils.scorer import setup_scorer,compute_score,scorers_runner
from utils.logger import setup_logger
from utils.qgg_optimizer import setup_optim,optims_runner
from utils.scheduler import setup_scheduler,step_scheduler
from utils.data_process import separate_answer_and_question
args = get_args()

def _parse_question(question):
    """
    Args:
        question: str
    Return:
        level,question
    """
    level = None
    try:
        level = re.match("\\[.*\\]",question).group()
    except:
        pass
    
    if level is not None:
        question = question.replace(level,"")
    return level,question

class Model(pl.LightningModule):
    def __init__(self,args=args):
        super().__init__()
        self.save_hyperparameters(args)
        args = get_args()
        self.hparams = args
        self.tokenizer = get_tokenizer()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids,attention_mask,labels=None):
        return self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels,return_dict=True)
    
    @step_scheduler
    def training_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1],batch[2])
        loss = outputs['loss']
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1],batch[2])
        loss = outputs['loss']
        self.log('dev_loss',loss)
    
    @setup_optim
    @setup_logger
    @setup_scorer
    def on_test_epoch_start(self):
        pass
    
    def test_step(self, batch, batch_idx):
        # tensor
        dataset_name = batch[0][0]
        input_ids = batch[1]
        attention_mask = batch[2]
        # string
        label_questions = batch[3]
        label_questions = [_q[0] for _q in label_questions]
        
        article = batch[4]
        article = article[0]

        input_ids_len = input_ids.shape[-1]
        batch_size = input_ids.shape[0]
        assert batch_size == 1

        num_return_sequences = args.gen_n
        sample_outputs = self.model.generate(
            input_ids = input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LENGTH,
            early_stopping=True,
            num_beams=num_return_sequences + 2,
            no_repeat_ngram_size=5,
            num_return_sequences=num_return_sequences
        )


        # assert len(sample_outputs) == num_return_sequences # 1
        decode_questions = ""
        sample_output = sample_outputs[0]     
        for sample_output in sample_outputs:
            decode_questions +=  "_$[0]" + self.tokenizer.decode(sample_output, skip_special_tokens=True)
    
        decode_questions = decode_questions.split('_$')
        new_decode_questions = []
        levels = []
        for decode_question in decode_questions:
            level,question = _parse_question(decode_question)
            if question =="": continue
            new_decode_questions.append(question)
            levels.append(level)
        decode_questions = new_decode_questions
        
        if args.dev: print(decode_questions)

        # clean qa pair format
        # the order of training target is `answer` -> `question`
        # but we changed to `question` -> `answer` here for readability
        decode_questions = [separate_answer_and_question(qa) for qa in decode_questions]
        decode_questions = [f"{qa['question_text']} {qa['answer_text']}" for qa in decode_questions]

        label_questions = [separate_answer_and_question(qa) for qa in label_questions]
        label_questions = [f"{qa['question_text']} {qa['answer_text']}" for qa in label_questions]

        optims_results = optims_runner(
            optims=self.qgg_optimizers,
            optim_names=args.qgg_optims,
            condicate_questions=decode_questions,
            context=article
        )

        scorers_runner(
            scoers=self.scorers,
            optim_names=args.qgg_optims,
            optims_results=optims_results,
            label_questions=label_questions,
            article=article,
            predict_logger = self.predict_logger
        )

    @compute_score
    def test_epoch_end(self,outputs):
        pass
    
    @setup_scheduler
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=args.lr)
