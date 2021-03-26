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
# from utils import compute_coverage_score
from utils.scorer import SimilarityScorer,CoverageScorer
from utils.logger import PredictLogger

args = get_args()

class CustomMixin():
    def feedback_generation(self, input_ids, feedback_times = 3):
        outputs = []
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        #
        input_ids = input_ids.squeeze(0).tolist()        
        # gen_ids = None

        for i in range(feedback_times):            
            gened_ids = self.tokenizer(GENED_TOKEN + self.tokenizer.sep_token.join(outputs) + GENED_TOKEN,add_special_tokens=False)['input_ids']
            input_ids = gened_ids + input_ids
            input_ids = input_ids[:MAX_LENGTH]
            
            sample_outputs = self.model.generate(
                input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device),
                attention_mask=torch.LongTensor([1]*len(input_ids)).unsqueeze(0).to(device),
                max_length=MAX_LENGTH,
                early_stopping=True,
                temperature=0.85,
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
            if WARN_UP_TOKEN != "":
                decode_questions = decode_questions.replace(WARN_UP_TOKEN,"")
            if self.tokenizer.bos_token is not None:
                decode_questions = re.sub(re.escape(self.tokenizer.bos_token),'',decode_questions)
            decode_questions = decode_questions.strip()
            if args.dev: print(decode_questions)
            outputs.append(decode_questions)
        return outputs

class Model(pl.LightningModule,CustomMixin):
    def __init__(self,args=args):
        super().__init__()
        self.save_hyperparameters(args)

        #
        args = get_args()
        self.tokenizer = get_tokenizer()
        self.model = CustomBartForConditionalGeneration.from_pretrained(args.base_model)
        self.model.resize_token_embeddings(len(self.tokenizer))
        # self.automatic_optimization = False
        self.opt = torch.optim.AdamW(self.parameters(), lr=args.lr)

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

        if args.disable_negative_loss == False: # use negative_loss
            self.log_dict({'n_loss': n_loss}, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('dev_loss',loss,prog_bar=True)
    
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

        batch_size = input_ids.shape[0]
        assert batch_size == 1

        decode_questions = self.feedback_generation(input_ids,feedback_times=args.gen_n)
        
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
        return self.opt
