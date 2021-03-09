import pytorch_lightning as pl
# from custom_transformers.src.transformers import BartForConditionalGeneration
from .modeling_bart import CustomBartForConditionalGeneration
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
import re
import os
import json
from .config import *
from utils import compute_coverage_score


args = get_args()

class CustomMixin():
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
    
    def feedback_generation(self, input_ids, feedback_times = 3):
        outputs = []
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        #
        input_ids = input_ids.squeeze(0).tolist()        
        gen_ids = None

        for i in range(feedback_times):
            if gen_ids is not None:
                input_ids = gen_ids + input_ids
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
                num_beams=3,
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
            decode_questions = re.sub('^'+re.escape('_$'),'',decode_questions)
            if args.dev: print(decode_questions)
            gen_ids = self.tokenizer("_$[0]"+decode_questions, max_length=MAX_LENGTH, truncation=True, add_special_tokens=False)['input_ids']
            outputs.append(decode_questions[3:])
        return outputs

class Model(pl.LightningModule,CustomMixin):
    def __init__(self):
        super().__init__()
        self.tokenizer = get_tokenizer()
        self.model = CustomBartForConditionalGeneration.from_pretrained(args.base_model)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids,attention_mask,labels=None,negative_sample_ids=None):
        return self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels,return_dict=True,negative_sample_ids=negative_sample_ids)
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids = batch[0],
            attention_mask = batch[1],
            labels = batch[2],
            negative_sample_ids = batch[3]
            )
        loss = outputs['loss']
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(input_ids = batch[0], attention_mask = batch[1], labels = batch[2], negative_sample_ids = batch[3])
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

        batch_size = input_ids.shape[0]
        assert batch_size == 1

        decode_questions = self.feedback_generation(input_ids)
        
        output =  {
            'batch_idx':batch_idx,
            'dataset_name':dataset_name,
            'questions':decode_questions,
            'labels':[_q[0] for _q in label_questions],
            'article':article[0],
            'levels': ['[0]']*len(decode_questions)
        }


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
