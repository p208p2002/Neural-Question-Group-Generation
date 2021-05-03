import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
import re
import os
import json
from .config import *
from utils.data_process import process_decode_questions
from utils.scorer import setup_scorer,compute_score
from utils.logger import setup_logger
from utils.qgg_optimizer import setup_optim
from utils.scheduler import step_scheduler,setup_scheduler
from loguru import logger

args = get_args()

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

        num_return_sequences = 1
        sample_outputs = self.model.generate(
            input_ids = input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LENGTH,
            early_stopping=True,
            temperature=0.85,
            do_sample=True,
            top_p=0.9,
            top_k=10,          
            num_beams=1,
            no_repeat_ngram_size=5,
            num_return_sequences=num_return_sequences,
            
        )

        assert len(sample_outputs) == num_return_sequences # 1
        sample_output = sample_outputs[0]        
        decode_questions = self.tokenizer.decode(sample_output, skip_special_tokens=False)
        if args.dev: print(decode_questions)
        decode_questions = re.sub(re.escape(self.tokenizer.pad_token),'',decode_questions)
        decode_questions = re.sub(re.escape(self.tokenizer.eos_token),'',decode_questions)
        if self.tokenizer.bos_token is not None:
            decode_questions = re.sub(re.escape(self.tokenizer.bos_token),'',decode_questions)
        decode_questions = decode_questions.strip()
        decode_questions = re.sub("^"+re.escape(SEP_TOKEN),'',decode_questions)
        decode_questions = decode_questions.split(SEP_TOKEN)
        decode_questions = decode_questions[:args.gen_n]

        #
        decode_questions = process_decode_questions(
            article = article,
            label_questions = label_questions,
            decode_questions = decode_questions,
            args = args,
            qgg_optimizers= self.qgg_optimizers,
            scorers = self.scorers,
            predict_logger = self.predict_logger
        )
        
    @compute_score
    def test_epoch_end(self,outputs):
        pass

    @setup_scheduler
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=args.lr)
