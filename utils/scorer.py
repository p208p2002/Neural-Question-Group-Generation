from nlgeval import NLGEval
from collections import defaultdict
import os
import re
import stanza

def setup_scorer(func):
    def wrapper(*args,**kwargs):
        self = args[0]
        self.reference_scorer = SimilarityScorer()
        self.classmate_scorer = SimilarityScorer()
        self.keyword_coverage_scorer = CoverageScorer()
        return func(*args,**kwargs)
    return wrapper

def compute_score(func):
    def wrapper(*args,**kwargs):
        self = args[0]
        self._log_dir = os.path.join(self.trainer.default_root_dir,'dev') if self.trainer.log_dir is None else self.trainer.log_dir
        self.reference_scorer.compute(save_report_dir=self._log_dir,save_file_name='reference_score.txt')
        self.classmate_scorer.compute(save_report_dir=self._log_dir,save_file_name='classmate_score.txt')
        self.keyword_coverage_scorer.compute(save_report_dir=self._log_dir,save_file_name='keyword_coverage_score.txt')
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
    
    def __del__(self):
        del self.nlgeval
        if self.preprocess:
            del self.nlp
    
    def _preprocess(self,raw_sentence):
        result = self.nlp(raw_sentence.replace("\n\n",""))
        tokens = []
        try:
            for token in result.sentences[0].tokens:
                tokens.append(token.text.lower())
                tokenize_sentence = ' '.join(tokens)
        except:
            print('_preprocess fail, return ""\n',raw_sentence,result)
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
    def add(self,hyp,refs):
        refs = refs[:]
        if self.preprocess:
            hyp = self._preprocess(hyp)
            refs = [self._preprocess(ref) for ref in refs]
        _score = self.nlgeval.compute_individual_metrics(hyp=hyp, ref=refs)
        for score_key in _score.keys():
            self.score[score_key] += _score[score_key]
        self.len += 1

class CoverageScorer(Scorer):
    def __init__(self,preprocess=True):
        super().__init__(preprocess=preprocess)
        self.stop_words_en = open('/user_data/MasterQG/utils/stopwords-en.txt','r',encoding='utf-8')
        self.stop_words_sign = open('/user_data/MasterQG/utils/stopwords-sign.txt','r',encoding='utf-8')
        self.stop_words = self.stop_words_en.read().split() + self.stop_words_sign.read().split()
    
    def __del__(self):
        self.stop_words_en.close()
        self.stop_words_sign.close()

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