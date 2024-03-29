import pytorch_lightning as pl
# from custom_transformers.src.transformers import BartForConditionalGeneration
from .modeling_bart import CustomBartForConditionalGeneration
from .tokenizer import get_tokenizer,GENED_TOKEN
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
from utils import save_huggingface_model
from loguru import logger
import time

args = get_args()

class Model(pl.LightningModule):
    def __init__(self,args=args):
        super().__init__()
        self.save_hyperparameters(args)

        #
        args = get_args()
        self.hparams = args
        self.tokenizer = get_tokenizer()
        self.model = CustomBartForConditionalGeneration.from_pretrained(args.base_model)
        self.model.resize_token_embeddings(len(self.tokenizer))        
        
    def forward(self, input_ids,attention_mask,labels=None,use_negative_loss=False,decoder_input_ids=None):
        return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids = None,
                labels=labels,
                return_dict=True,
                use_negative_loss=use_negative_loss
            )
    
    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items["loss"] = items.pop("loss", None) # change display order
        return items
    
    # @step_scheduler
    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids = batch[0],
            attention_mask = batch[1],
            decoder_input_ids = batch[2],
            labels = batch[3],
            use_negative_loss = False
            )
        loss = outputs['loss']

        if args.disable_negative_loss == False: # use negative_loss
            labels = batch[2]
            n_labels = batch[5]
            # n_labels = torch.where(labels == n_labels,torch.LongTensor([-100]).to(n_labels.device),n_labels)

            n_outputs = self(
                input_ids = batch[0],
                attention_mask = batch[1],
                decoder_input_ids = batch[4],
                labels = n_labels,
                use_negative_loss = True
                )
            n_loss = n_outputs['loss']
            loss += n_loss
            self.log('n_loss',n_loss,prog_bar=True)
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('dev_loss',loss,prog_bar=True)
    
    @save_huggingface_model
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

        batch_size = input_ids.shape[0]
        assert batch_size == 1

        decode_questions = self.feedback_generation(input_ids,feedback_times=args.gen_n)
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
    
    # @setup_scheduler
    def configure_optimizers(self):
        self.opt = torch.optim.AdamW(self.parameters(), lr=args.lr)
        return self.opt
    
    def feedback_generation(self, input_ids, feedback_times = 3):
        outputs = []
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        #
        input_ids = input_ids.squeeze(0).tolist()        
        # gen_ids = None

        for i in range(feedback_times):
            gened_text = self.tokenizer.bos_token * (len(outputs)+1)
            gened_ids = self.tokenizer(gened_text,add_special_tokens=False)['input_ids']            
            input_ids = gened_ids + input_ids
            input_ids = input_ids[:MAX_LENGTH]
            
            sample_outputs = self.model.generate(
                input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device),
                attention_mask=torch.LongTensor([1]*len(input_ids)).unsqueeze(0).to(device),
                max_length=MAX_LENGTH,
                early_stopping=True,
                temperature=1.0,
                do_sample=True,
                top_p=0.9,
                top_k=10,
                num_beams=1,
                no_repeat_ngram_size=5,
                num_return_sequences=1,
            )
            sample_output = sample_outputs[0]        
            decode_questions = self.tokenizer.decode(sample_output, skip_special_tokens=False)
            decode_questions = re.sub(re.escape(self.tokenizer.pad_token),'',decode_questions)
            decode_questions = re.sub(re.escape(self.tokenizer.eos_token),'',decode_questions)
            if self.tokenizer.bos_token is not None:
                decode_questions = re.sub(re.escape(self.tokenizer.bos_token),'',decode_questions)
            decode_questions = decode_questions.strip()
            # if args.dev: print(decode_questions)
            outputs.append(decode_questions)
        return outputs
