import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from .scorer import CoverageScorer,SimilarityScorer
import re
import random

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
        self.similarity_scorer = SimilarityScorer()
        self.model=ga(
            function=self.fitness_function,
            dimension=candicate_pool_size,
            variable_type='bool',
            convergence_curve=True,
            algorithm_parameters = {
                'max_num_iteration': 32,
                'population_size':12,
                'mutation_probability':0.15,
                'elit_ratio': 0.01,
                'crossover_probability': 0.5,
                'parents_portion': 0.5,
                'crossover_type':'uniform',
                'max_iteration_without_improv':8
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
        classmate_similarity_score = self.similarity_scorer.compute(return_score=True).get('Bleu_2',0.0)
        
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
            keyword_coverage_score*0.45\
            + (1-classmate_similarity_score)*0.45\
            + diversity_score*0.1
                
        # punishment if count_pick not equal to question_group_size
        count_pick = (genome==True).sum()
        if count_pick != self.target_question_qroup_size:
            score = score*0.5

        return score*-1
    
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

# class GreedyOptimizer():