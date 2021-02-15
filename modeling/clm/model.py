import pytorch_lightning as pl
from transformers import AutoModelForCausalLM
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
import re
import os
import json
from .config import *
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
        self.model = AutoModelForCausalLM.from_pretrained(args.base_model)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids,attention_mask,labels=None):
        return self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels,return_dict=True)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1],batch[2])
        return outputs['loss']
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1],batch[2])
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
            attention_mask = attention_mask,
            max_length=MAX_LENGTH,
            early_stopping=True,
            temperature=0.8,
            do_sample=True,
            top_p=0.80,
            top_k=10,
            no_repeat_ngram_size=4,
            num_return_sequences=num_return_sequences,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        assert len(sample_outputs) == num_return_sequences # 1
        sample_output = sample_outputs[0]        
        
        decode_questions = self.tokenizer.decode(sample_output[input_ids_len:], skip_special_tokens=False)
        decode_questions = re.sub(re.escape(self.tokenizer.pad_token),'',decode_questions)
        
        if 'm_race' in args.datasets:
            decode_questions = decode_questions.split('_$')            
            new_decode_questions = []
            levels = []
            for decode_question in decode_questions:
                level,question = _parse_question(decode_question)
                new_decode_questions.append(question)
                levels.append(level)
            decode_questions = new_decode_questions
        else:
            decode_questions = decode_questions.split(self.tokenizer.sep_token)

        if decode_questions[-1] == self.tokenizer.eos_token:
            decode_questions.pop(-1)
        
        if decode_questions[0] == '':decode_questions.pop(0)

        output =  {
            'batch_idx':batch_idx,
            'dataset_name':dataset_name,
            'questions':decode_questions,
            'labels':[_q[0] for _q in label_questions],
            'article':article[0]
        }
        
        if 'm_race' in args.datasets:            
            output['levels'] = levels[1:]

        # add score
        output['question_scores'] = []
        for question in output['questions']:
            score = self.nlgeval.compute_individual_metrics(hyp=question, ref=output['labels'])
            del score['CIDEr']
            bP, bR, bF1 = self.bert_scorer.score([question], [output['labels']])
            score['BertScore'] = bF1.item()
            for k in score.keys(): score[k] = str(score[k])
            output['question_scores'].append(score)

        # log
        log_dir = os.path.join(self.trainer.default_root_dir,'dev') if self.trainer.log_dir is None else self.trainer.log_dir
        os.makedirs(log_dir,exist_ok=True)
        with open(os.path.join(log_dir,'predict.jsonl'),'a',encoding='utf-8') as log_f:
            output_str = json.dumps(output,ensure_ascii=False) + '\n'
            log_f.write(output_str)
                
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=args.lr)
