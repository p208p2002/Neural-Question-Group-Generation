from nlgeval import NLGEval
from collections import defaultdict
import os
import re
import stanza
from loguru import logger
import copy
from functools import lru_cache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_scorer(func):
    def wrapper(*_args,**_kwargs):
        self = _args[0]
        args = self.hparams
        
        # we set scoers for each optim
        self.scorers = []
        count_qgg_optims = len(args.qgg_optims)
        for i in range(count_qgg_optims):
            self.scorers.append({
                'reference_scorer':SimilarityScorer(),
                'classmate_scorer':SimilarityScorer(),
                'keyword_coverage_scorer':CoverageScorer()
            })

        return func(*_args,**_kwargs)
    return wrapper

def scorers_runner(scoers,optim_names,optims_results,label_questions,article,predict_logger):
    assert len(scoers) == len(optim_names)
    assert len(scoers) == len(optims_results),f"scoers:{scoers}, optims_results:{optims_results}"

    _log_dict = {
        'article':article,
        'label_questions':label_questions,
    }

    for scorer,optim_name,decode_questions in zip(scoers,optim_names,optims_results):
        # save optims_results for log
        _log_dict[optim_name] = copy.deepcopy(decode_questions)

        #
        reference_scorer = scorer['reference_scorer']
        classmate_scorer = scorer['classmate_scorer']
        keyword_coverage_scorer = scorer['keyword_coverage_scorer']

        # reference socre
        for decode_question in decode_questions:
            reference_scorer.add(hyp=decode_question,refs=label_questions)
        
        # classmate score
        if len(decode_questions) > 1:
            for decode_question in decode_questions[:]:
                classmate_questions = decode_questions[:]
                classmate_questions.remove(decode_question)
                classmate_scorer.add(hyp=decode_question,refs=classmate_questions)
        # keyword coverage score
        keyword_coverage_scorer.add(decode_questions,article)
    
    # wirte log
    predict_logger.log(_log_dict)

def compute_score(func):
    def wrapper(*args,**kwargs):
        self = args[0]
        self._log_dir = os.path.join(self.trainer.default_root_dir,'dev') if self.trainer.log_dir is None else self.trainer.log_dir
        assert len(self.scorers) == len(self.hparams.qgg_optims)
        for scorer,opt_name in zip(self.scorers,self.hparams.qgg_optims):
            #
            reference_scorer = scorer['reference_scorer']
            classmate_scorer = scorer['classmate_scorer']
            keyword_coverage_scorer = scorer['keyword_coverage_scorer']

            #
            reference_scorer.compute(save_report_dir=os.path.join(self._log_dir,opt_name),save_file_name='reference_score.txt')
            classmate_scorer.compute(save_report_dir=os.path.join(self._log_dir,opt_name),save_file_name='classmate_score.txt')
            keyword_coverage_scorer.compute(save_report_dir=os.path.join(self._log_dir,opt_name),save_file_name='keyword_coverage_score.txt')
        return func(*args,**kwargs)
    return wrapper

class Scorer():
    def __init__(self,preprocess=True,metrics_to_omit=["CIDEr"]):
        self.preprocess = preprocess
        self.nlgeval = NLGEval(no_glove=True,no_skipthoughts=True,metrics_to_omit=metrics_to_omit)
        self.score = defaultdict(lambda : 0.0)
        self.len = 0
        if self.preprocess:
            self.nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True, verbose=False)
        
        #
        self.stop_words_sign = open('utils/stopwords-sign.txt','r',encoding='utf-8').read().split()
        self.stop_words_sign_rule = "|".join([re.escape(sign) for sign in self.stop_words_sign])
    
    def __del__(self):
        # del self.nlgeval
        if self.preprocess:
            del self.nlp
    
    @lru_cache(maxsize=200)
    def _preprocess(self,raw_sentence):
        result = self.nlp(raw_sentence.replace("\n\n",""))
        tokens = []
        try:
            for token in result.sentences[0].tokens:
                tokens.append(token.text.lower())
            tokenize_sentence = ' '.join(tokens)
            tokenize_sentence = re.sub(self.stop_words_sign_rule,"",tokenize_sentence)                
        except Exception as e:
            logger.warning(e)
            logger.warning(f'preprocess fail, return "" raw_sentence:{raw_sentence} result:{result}')
            return ""
        return tokenize_sentence
    
    def clean(self):
        self.score = defaultdict(lambda : 0.0)
        self.len = 0
    
    def add(*args,**kwargs):
        assert False,'no implement error'

    def compute(self,save_report_dir=None,save_file_name='score.txt',return_score=False):
        # 
        out_score = {}
        
        if save_report_dir is not None:
            os.makedirs(save_report_dir,exist_ok=True)
            save_score_report_path = os.path.join(save_report_dir,save_file_name)
            score_f = open(save_score_report_path,'w',encoding='utf-8')
        for score_key in self.score.keys():
            _score = self.score[score_key]/self.len
            out_score[score_key] = _score
            if save_report_dir is not None:
                score_f.write("%s\t%3.5f\n"%(score_key,_score))
        
        if return_score:
            return out_score
    
class SimilarityScorer(Scorer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ppl_scorer = PPLScorer()

    def add(self,hyp,refs):
        refs = refs[:]
        if self.preprocess:
            hyp = self._preprocess(hyp)
            refs = [self._preprocess(ref) for ref in refs]
        _score = self.nlgeval.compute_individual_metrics(hyp=hyp, ref=refs)
        ppl_score = self.ppl_scorer._compute_ppl(hyp)
        scaled_ppl = self.ppl_scorer._compute_scaled_ppl(hyp)
        _score['ppl'] = ppl_score
        _score['scaled_ppl'] = scaled_ppl
        for score_key in _score.keys():
            self.score[score_key] += _score[score_key]
        self.len += 1

class CoverageScorer(Scorer):
    def __init__(self,preprocess=True):
        super().__init__(preprocess=preprocess)
        self.stop_words_en = open('utils/stopwords-en.txt','r',encoding='utf-8')
        self.stop_words_sign = open('utils/stopwords-sign.txt','r',encoding='utf-8')
        self.stop_words = self.stop_words_en.read().split() + self.stop_words_sign.read().split()
        
        # some sign used to split context to sentence, remove them from `stopwords-sign`
        self.stop_words_sign = open('utils/stopwords-sign.txt','r',encoding='utf-8').read().split()
        self.stop_words_sign.remove(',')
        self.stop_words_sign.remove('.')
        self.stop_words_sign.remove('!')
        self.stop_words_sign.remove('?')
        self.stop_words_sign_rule = "|".join([re.escape(sign) for sign in self.stop_words_sign])
    
    # def __del__(self):
        # self.stop_words_en.close()
        # self.stop_words_sign.close()

    def _compute_coverage_score(self,sents:list,article:str):
        sent = ' '.join(sents)
        sent_list = re.split(r",|\.|\!|\?",sent)
        for sent in sent_list[:]:
            if sent == '': sent_list.remove(sent)
        
        # get sents keywords
        keyword_list = []
        for sent in sent_list[:]:
            sent = sent.lower()
            word_list = sent.split()
            for word in word_list:
                if word not in self.stop_words:
                    keyword_list.append(word)
        
        # process acticle into words and compute coverage
        article_sent_list = re.split(r",|\.|\!|\?",article)
        
        count_article_sent = len(article_sent_list)
        if count_article_sent == 0:
            return 0.0
        
        count_coverage = 0
        for article_sent in article_sent_list:
            article_sent = article_sent.lower().split()
            for keyword in keyword_list:
                if keyword in article_sent:
                    count_coverage += 1
                    break 
        return count_coverage/count_article_sent
        
    def add(self,sents:list,article:str):
        sents = sents[:]
        if self.preprocess:
            sents = [self._preprocess(sent) for sent in sents]
            article = self._preprocess(article)
        coverage_score = self._compute_coverage_score(sents,article)
        self.score['keyword_coverage'] += coverage_score
        self.len += 1

class PPLScorer(Scorer):
    def __init__(self, model_id = 'gpt2', device = 'cpu', stride=512, max_length=512):
        if '_ppl_model' not in globals():
            global _ppl_model
            _ppl_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.model = _ppl_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.stride = stride
        self.max_length = max_length
        self.device = device
        
        #
        self.score = defaultdict(lambda : 0.0)
        self.len = 0
    
    def add(self,sentence):
        self.score['ppl'] += self._compute_ppl(sentence)
        self.len += 1
        
    def _compute_scaled_ppl(self,sentence,alpha=0.2):
        # https://www.desmos.com/calculator/scqyyq0ody
        avg_ll = self._compute_avg_log_likelihood(sentence)
        return torch.exp(-avg_ll*alpha)
    
    def _compute_ppl(self,sentence):
        # https://huggingface.co/transformers/perplexity.html
        avg_ll = self._compute_avg_log_likelihood(sentence)
        return torch.exp(avg_ll)
        
    
    @lru_cache(maxsize=200)
    def _compute_avg_log_likelihood(self,sentence):
        stride = self.stride
        max_length = self.max_length
        encodings = self.tokenizer(sentence, return_tensors='pt')
        model = self.model

        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i    # may be different from stride on last loop
            input_ids = encodings.input_ids[:,begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:,:-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)                
                log_likelihood = outputs[0] * trg_len

            lls.append(log_likelihood)
        return torch.stack(lls).sum() / end_loc