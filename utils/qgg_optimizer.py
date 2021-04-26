import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from .scorer import CoverageScorer,SimilarityScorer
import re
import random
from loguru import logger
from utils import timer

def setup_optim(func):
    def wrapper(*_args,**_kwargs):
        self = _args[0]
        args = self.hparams
        self.qgg_optimizers = []
        assert len(args.qgg_optims) >= 1,'must spec at least one qgg_optim'
        for qgg_optim in args.qgg_optims:
            if qgg_optim == 'ga':
                self.qgg_optimizers.append(GAOptimizer(candicate_pool_size=args.gen_n,target_question_qroup_size=args.pick_n))
            elif qgg_optim == 'random':
                self.qgg_optimizers.append(RandomOptimizer(candicate_pool_size=args.gen_n,target_question_qroup_size=args.pick_n))
            elif qgg_optim == 'first-n':
                self.qgg_optimizers.append(FirstNOptimizer(candicate_pool_size=args.gen_n,target_question_qroup_size=args.pick_n))
            elif qgg_optim == 'greedy':
                self.qgg_optimizers.append(GreedyOptimizer(candicate_pool_size=args.gen_n,target_question_qroup_size=args.pick_n))
            else:
                raise Exception("no qgg_optim match")
        return func(*_args,**_kwargs)
    return wrapper

def optims_runner(optims,optim_names,condicate_questions,context):
    results = []
    for optim,opt_name in zip(optims,optim_names):
        result = optim.optimize(condicate_questions=condicate_questions,context=context)
        # logger.debug(f"opt_name:{opt_name}, result:{result}")
        results.append(result)
    return results

class GAOptimizer():
    def __init__(self,candicate_pool_size,target_question_qroup_size):
        """
        Args:
            candicate_pool_size: how many question in the candicate pool, refs to encoding size
            target_question_qroup_size: the questions number we execpt to pick
        """
        assert target_question_qroup_size <= candicate_pool_size,'candicate_pool_size should smaller than target_question_qroup_size'
        self.target_question_qroup_size = target_question_qroup_size
        self.condicate_questions = None
        self.context = None
        self.coverage_scorer = CoverageScorer()
        self.similarity_scorer = SimilarityScorer(metrics_to_omit=["CIDEr","METEOR",'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4'])
        self.candicate_pool_size = candicate_pool_size
        self.model=ga(
            function=self.fitness_function,
            dimension=candicate_pool_size,
            variable_type='bool',
            convergence_curve=False,
            algorithm_parameters = {
                'max_num_iteration': 40,
                'population_size':20,
                'mutation_probability':0.2,
                'elit_ratio': 0.2,
                'crossover_probability': 0.6,
                'parents_portion': 0.5,
                'crossover_type':'two_point',
                'max_iteration_without_improv':10
            }
        )
    
    def fitness_function(self,genome):
        # reset scorer
        self.coverage_scorer.clean()
        self.similarity_scorer.clean()
        
        #
        pick_questions = self.decode(genome)
        
        # question group keyword_coverage_score
        self.coverage_scorer.add(pick_questions,self.context)
        keyword_coverage_score = self.coverage_scorer.compute(return_score=True)['keyword_coverage']
        
        # question qroup classmate_similarity_score
        if len(pick_questions) > 1:
            for pick_question in pick_questions[:]:
                classmate_questions = pick_questions[:]
                classmate_questions.remove(pick_question)
                self.similarity_scorer.add(hyp=pick_question,refs=classmate_questions)
        classmate_similarity_score = self.similarity_scorer.compute(return_score=True).get('ROUGE_L',0.0)
        
        # diversity_score
        has_s_type = False
        has_c_type = False
        diversity_score = 0
        for pick_question in pick_questions:
            if re.search(re.escape("?")+"$",pick_question):
                has_s_type = True
            if re.search(re.escape("_"),pick_question):
                has_c_type = True
        if has_s_type and has_c_type:
            diversity_score = 1
        elif has_s_type or has_c_type:
            diversity_score = 0.5
    
        score = \
            keyword_coverage_score\
            + (1-classmate_similarity_score)\
            + diversity_score
                
        # punishment if count_pick not equal to question_group_size
        count_pick = (genome==True).sum()
        punish_weight = 1 - (abs(self.target_question_qroup_size - count_pick)/self.candicate_pool_size)
        return score*punish_weight*-1
    
    def decode(self,genome):
        pick_questions = []
        for p_id,is_pick in enumerate(genome):
            is_pick = bool(is_pick)
            if is_pick:
                pick_questions.append(self.condicate_questions[p_id][:])
        return pick_questions
    
    def optimize(self,condicate_questions,context,*args,**kwargs):
        """
        Args:
            condicate_questions: the condicate questions
            context: context that used to gen condicate questions
        """
        while(len(condicate_questions) < self.candicate_pool_size):
            condicate_questions.append("")
            print("Warning!!: `len(condicate_questions) < self.candicate_pool_size`")

        self.context = context
        self.condicate_questions = condicate_questions
        self.model.run()
        return self.decode(self.model.best_variable)

class RandomOptimizer():
    def __init__(self,candicate_pool_size,target_question_qroup_size):
        """
        Args:
            candicate_pool_size: how many question in the candicate pool, refs to encoding size
            target_question_qroup_size: the questions number we execpt to pick
        """
        assert target_question_qroup_size <= candicate_pool_size,'candicate_pool_size should smaller than target_question_qroup_size'
        self.target_question_qroup_size = target_question_qroup_size

    def optimize(self,condicate_questions,*args,**kwargs):
        condicate_questions = condicate_questions[:]
        random.shuffle(condicate_questions)
        return condicate_questions[:self.target_question_qroup_size]

class FirstNOptimizer():
    def __init__(self,candicate_pool_size,target_question_qroup_size):
        """
        Args:
            candicate_pool_size: how many question in the candicate pool, refs to encoding size
            target_question_qroup_size: the questions number we execpt to pick
        """
        assert target_question_qroup_size <= candicate_pool_size,'candicate_pool_size should smaller than target_question_qroup_size'
        self.target_question_qroup_size = target_question_qroup_size

    def optimize(self,condicate_questions,*args,**kwargs):
        return condicate_questions[:self.target_question_qroup_size]

class GreedyOptimizer():
    def __init__(self,candicate_pool_size,target_question_qroup_size):
        """
        Args:
            candicate_pool_size: how many question in the candicate pool, refs to encoding size
            target_question_qroup_size: the questions number we execpt to pick
        """
        assert target_question_qroup_size <= candicate_pool_size,'candicate_pool_size should smaller than target_question_qroup_size'
        self.target_question_qroup_size = target_question_qroup_size
        self.condicate_questions = None
        self.context = None
        self.coverage_scorer = CoverageScorer()
        self.similarity_scorer = SimilarityScorer(metrics_to_omit=["CIDEr","METEOR",'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4'])

    def optimize(self,condicate_questions,context,*args,**kwargs):
        """
        Args:
            condicate_questions: the condicate questions
            context: context that used to gen condicate questions
        """
        question_with_score = {}
        for question in condicate_questions:
            # keyword_coverage
            self.coverage_scorer.clean()
            self.coverage_scorer.add([question],context)
            keyword_coverage_score = self.coverage_scorer.compute(return_score=True)['keyword_coverage']
            
            # classmate_similarity_score
            self.similarity_scorer.clean()
            classmate_questions = condicate_questions[:]
            classmate_questions.remove(question)
            if len(classmate_questions) > 1:
                self.similarity_scorer.add(hyp=question,refs=classmate_questions)
            classmate_similarity_score = self.similarity_scorer.compute(return_score=True).get('ROUGE_L',0.0)
            
            score = keyword_coverage_score + (1.0-classmate_similarity_score)
            question_with_score[question] = score
            
        question_scores = sorted(question_with_score.items(), key=lambda x:x[-1],reverse=True)
        return [q[0] for q in question_scores][:self.target_question_qroup_size]
    