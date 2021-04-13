import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
import re
import os
import json
from .config import *
from utils.scorer import setup_scorer,compute_score
from utils.logger import setup_logger
from utils.qgg_optimizer import setup_optimizer
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
        self.tokenizer = get_tokenizer()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids,attention_mask,labels=None):
        return self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels,return_dict=True)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1],batch[2])
        if args.base_model == 'transfo-xl-wt103':
            loss = outputs['losses'].mean()
        else:
            loss = outputs['loss']
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1],batch[2])
        if args.base_model == 'transfo-xl-wt103':
            loss = outputs['losses'].mean()
        else:
            loss = outputs['loss']
        self.log('dev_loss',loss)
    
    @setup_optimizer
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
    
        if 'm_race' in args.datasets:
            decode_questions = decode_questions.split('_$')
            new_decode_questions = []
            levels = []
            for decode_question in decode_questions:
                level,question = _parse_question(decode_question)
                if question =="": continue
                new_decode_questions.append(question)
                levels.append(level)
            decode_questions = new_decode_questions
        else:
            decode_questions = decode_questions.split(self.tokenizer.sep_token)
        
        if args.dev: print(decode_questions)
        
        decode_questions = self.qgg_optimizer.optimize(condicate_questions=decode_questions,context=article)

        # reference socre
        for decode_question in decode_questions:
            self.reference_scorer.add(hyp=decode_question,refs=label_questions)

        # classmate score
        if len(decode_questions) > 1:
            for decode_question in decode_questions[:]:
                classmate_questions = decode_questions[:]
                classmate_questions.remove(decode_question)
                self.classmate_scorer.add(hyp=decode_question,refs=classmate_questions)

        # keyword coverage score
        self.keyword_coverage_scorer.add(decode_questions,article)

        # predict log
        self.predict_logger.log({
            'article':article,
            'label_questions':label_questions,
            'decode_questions':decode_questions
        })

    @compute_score
    def test_epoch_end(self,outputs):
        pass
                
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=args.lr)
