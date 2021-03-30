import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
import re
import os
import json
from .config import *
from utils.scorer import SimilarityScorer, CoverageScorer
from utils.logger import PredictLogger

args = get_args()

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
        loss = outputs['loss']
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1],batch[2])
        loss = outputs['loss']
        self.log('dev_loss',loss)
    
    def on_test_epoch_start(self):
        self.reference_scorer = SimilarityScorer()
        self.classmate_scorer = SimilarityScorer()
        self.keyword_coverage_scorer = CoverageScorer()
        self._log_dir = os.path.join(self.trainer.default_root_dir,'dev') if self.trainer.log_dir is None else self.trainer.log_dir
        self.predict_logger = PredictLogger(save_dir=self._log_dir)
        
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
            max_length=int(MAX_LENGTH/2),
            early_stopping=True,
            temperature=0.85,
            do_sample=True,
            top_p=0.9,
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
        decode_questions = decode_questions.split(SEP_TOKEN)
        decode_questions = decode_questions[:args.gen_n]
        
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

    def test_epoch_end(self,outputs):
        self.reference_scorer.compute(save_report_dir=self._log_dir,save_file_name='reference_score.txt')
        self.classmate_scorer.compute(save_report_dir=self._log_dir,save_file_name='classmate_score.txt')
        self.keyword_coverage_scorer.compute(save_report_dir=self._log_dir,save_file_name='keyword_coverage_score.txt')
                
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=args.lr)
