from finchge.grammar.mapper import GenotypeMapper
from finchge.core import GrammaticalEvolution
from finchge.fitness.fitness_functions import StringMatchFitness
from finchge.fitness.fitness_evaluator import FitnessEvaluator
from finchge.utils.logger import ExperimentLogger
from finchge.config import FinchConfig, Keys
from finchge.grammar import Grammar
from finchge.algorithm import GeneticAlgorithm
from finchge.operators.selection import TournamentSelection
from finchge.operators.replacement import GenerationalReplacement
from finchge.operators.crossover import OnePointCrossover
from finchge.operators.mutation import IntFlipMutation
 


if __name__ == '__main__':
    # Load experiment configuration
    ge_config = FinchConfig.from_yaml('ge_config.yaml')

    # define fitness function
    fitness_fn = StringMatchFitness("hello")
    # Load grammar
    grammar = Grammar.from_file('grammar.bnf')
    ge_config = FinchConfig.from_yaml('ge_config.yaml')
    
    # Setup Mapper
    mapper = GenotypeMapper(grammar=grammar, 
                                max_wraps=ge_config.ge[Keys.MAX_WRAPS], 
                                max_recursion_depth=ge_config.ge[Keys.MAX_RECURSION_DEPTH])
    
    # Fitness evaluator handles the evaluation process
    fitness_evaluator = FitnessEvaluator(fitness_functions=fitness_fn,
                                         mapper=mapper,
                                         parallel_config=ge_config.parallel)

    # Setup Algorithm (if not provided Default parameters will be used by GrammaticalEvolution class)
    ga = GeneticAlgorithm(
            selection=TournamentSelection(max_best=False, tournament_size=ge_config.ge[Keys.TOURNAMENT_SIZE]),
            crossover=OnePointCrossover(codon_size=ge_config.ge[Keys.CODON_SIZE], 
                                        crossover_proba=ge_config.ge[Keys.CROSSOVER_PROBABILITY]),
            mutation=IntFlipMutation(ge_config.ge[Keys.MUTATION_PROBABILITY],
                                                            codon_size=ge_config.ge[Keys.CODON_SIZE]),
            replacement=GenerationalReplacement(max_best=False),
            elite_size=ge_config.ge[Keys.ELITE_SIZE],
            fitness_evaluator = fitness_evaluator
        )
    

    
    # Setup GE, the main experiment pipeline
    ge_ = GrammaticalEvolution(algorithm=ga, fitness_evaluator=fitness_evaluator)

 
    result = ge_.run() 
    print("Best Solution:", result.all_time_best.phenotype)
