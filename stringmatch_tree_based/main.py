import random

import numpy as np
from finchge.grammar.mapper import GenotypeMapper
from finchge.core import GrammaticalEvolution
from finchge.fitness.fitness_functions import StringMatchFitness
from finchge.fitness import FitnessEvaluator
from finchge.utils.logger import ExperimentLogger
from finchge.config import FinchConfig, Keys
from finchge.grammar import Grammar
from finchge.algorithm import GeneticAlgorithm
from finchge.operators.selection import TournamentSelection
from finchge.operators.crossover import SubtreeCrossover
from finchge.operators.mutation import SubtreeMutation
from finchge.operators.replacement import GenerationalReplacement

from finchge.grammar.tree_generator import TreeGenerator

import cProfile


if __name__ == '__main__': 
    # prepare fitness function
    fitness_fn = StringMatchFitness("hello")

    # load grammar and config
    grammar = Grammar.from_file('grammar.bnf')
    ge_config = FinchConfig.from_yaml('ge_config.yaml')

    # initialize tree generator
    tree_generator = TreeGenerator(grammar=grammar, max_tree_depth=ge_config.ge[Keys.MAX_TREE_DEPTH])

    mapper = GenotypeMapper(grammar=grammar, 
                                max_wraps=ge_config.ge[Keys.MAX_WRAPS], 
                                max_recursion_depth=ge_config.ge[Keys.MAX_RECURSION_DEPTH])
    

    fitness_evaluator = FitnessEvaluator(fitness_functions=fitness_fn, 
                                         mapper=mapper,
                                         parallel_config=ge_config.parallel
                                         )
    # Setup Tree based Operators
    subtree_crossover_=SubtreeCrossover(crossover_proba=ge_config.ge[Keys.CROSSOVER_PROBABILITY],
                                       non_terminals=grammar.non_terminals,
                                       tree_generator=tree_generator)
    subtree_mutation_=SubtreeMutation(tree_generator=tree_generator,
                                         non_terminals=grammar.non_terminals)
    
    ga = GeneticAlgorithm(
                selection=TournamentSelection(max_best=False, tournament_size=ge_config.ge[Keys.TOURNAMENT_SIZE]),
                crossover=subtree_crossover_,
                mutation=subtree_mutation_,
                replacement=GenerationalReplacement(max_best=False),
                elite_size=ge_config.ge[Keys.ELITE_SIZE],
                fitness_evaluator = fitness_evaluator
            ) 
    
    
    

    expt_logger = ExperimentLogger()
    # Setup GE
    ge_ = GrammaticalEvolution(algorithm=ga, fitness_evaluator=fitness_evaluator, expt_logger=expt_logger)

    #cProfile.run('ge_.find_fittest()')
    result = ge_.run()  
    print("Best Solution:", result.all_time_best.phenotype)
