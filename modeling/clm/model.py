import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
import re
import os
import json
from .config import *
from utils import compute_coverage_score
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
    def __init__(self):
        super().__init__()
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
    
    def on_test_epoch_start(self):
        #
        print("loading NLGEval...",end="\r")
        from nlgeval import NLGEval
        self.nlgeval = NLGEval(no_glove=True,no_skipthoughts=True)  # loads the models
        print("loading NLGEval...finish")

        #
        print("loading BERTScorer...",end="\r")
        import logging,os
        import transformers
        os.environ["TOKENIZERS_PARALLELISM"] = 'true'
        transformers.tokenization_utils.logger.setLevel(logging.ERROR)
        transformers.configuration_utils.logger.setLevel(logging.ERROR)
        transformers.modeling_utils.logger.setLevel(logging.ERROR)
        from bert_score import BERTScorer
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        print("loading BERTScorer...finish")
    
    def compute_score(self,hyp,refs):
        #
        hyp = hyp.strip().replace("\n","")
        if hyp == '': hyp = '#'

        refs = refs[:]
        refs = [ref.strip().replace("\n","") for ref in refs]
        for ref in refs[:]:
            if ref == '': refs.remove(ref)
        if len(refs) == 0: refs.append("@")


        # token scores    
        score = self.nlgeval.compute_individual_metrics(hyp=hyp, ref=refs)
        
        del score['CIDEr']

        # bert score
        bP, bR, bF1 = self.bert_scorer.score([hyp], [refs])
        score['BertScore'] = bF1.item() if bF1.item() > 0.0 else 0.0


        for k in score.keys(): score[k] = str(score[k])

        return score

    
    def test_step(self, batch, batch_idx):
        # tensor
        dataset_name = batch[0][0]
        input_ids = batch[1]
        attention_mask = batch[2]
        # string
        label_questions = batch[3]
        article = batch[4]

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
            # top_k=12,
            # num_beams=3,
            no_repeat_ngram_size=5,
            num_return_sequences=num_return_sequences,
            # bos_token_id=self.model.config.decoder_start_token_id,
            # eos_token_id=self.tokenizer.eos_token_id,
            # pad_token_id=self.tokenizer.pad_token_id
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
        decode_questions = re.sub('^'+re.escape('_$'),'',decode_questions)
        
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

        # if len(decode_questions) >0 and decode_questions[-1] == self.tokenizer.eos_token:
        #     decode_questions.pop(-1)
        # print(len(decode_questions))

        output =  {
            'batch_idx':batch_idx,
            'dataset_name':dataset_name,
            'questions':decode_questions,
            'labels':[_q[0] for _q in label_questions],
            'article':article[0]
        }

        if 'm_race' in args.datasets:            
            output['levels'] = levels
        

        # add score
        output['question_scores'] = []
        output['unlike_question_scores'] = []
        for i,question in enumerate(output['questions']):

            # like score
            score = self.compute_score(question,output['labels'])
            output['question_scores'].append(score)

            # unlike score
            questions = output['questions'][:]
            questions.pop(i)
            score = self.compute_score(hyp=question, refs=questions)
            output['unlike_question_scores'].append(score)
        
        #
        output['unlike_label_scores'] = []
        for i,label in enumerate(output['labels']):
            # unlike score
            labels = output['labels'][:]
            labels.pop(i)
            score = self.compute_score(hyp=label, refs=labels)
            output['unlike_label_scores'].append(score)

        output['question_coverage_score'] = compute_coverage_score(output['questions'],output['article'])
        output['label_coverage_score'] = compute_coverage_score(output['labels'],output['article'])

        # log
        log_dir = os.path.join(self.trainer.default_root_dir,'dev') if self.trainer.log_dir is None else self.trainer.log_dir
        os.makedirs(log_dir,exist_ok=True)
        with open(os.path.join(log_dir,'predict.jsonl'),'a',encoding='utf-8') as log_f:
            output_str = json.dumps(output,ensure_ascii=False) + '\n'
            log_f.write(output_str)
                
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=args.lr)
