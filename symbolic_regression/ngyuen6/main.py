
import numpy as np
from finchge.benchmarks.regression import NguyenBenchmark
from finchge.fitness.fitness_functions import RMSEFitness
from finchge.runners.sr import SymbolicRegressionRunner
from finchge.grammar import Grammar
from finchge.grammar.mapper import GenotypeMapper
from finchge.fitness import FitnessEvaluator
from finchge.core import GrammaticalEvolution
from finchge.config import FinchConfig, Keys
import pandas as pd 
from finchge.utils.logger import ExperimentLogger
 
if __name__ == '__main__':

    # Initialize the Nguyen benchmark problem (version 6)
    # This provides a standard symbolic regression test problem
    benchmark = NguyenBenchmark(
        version = 6,
        random_state=42,
        train_samples=20,
        test_samples=1000
    ) 

    # Generate training and testing datasets
    # X_train, y_train: data for evolving the symbolic expression
    # X_test, y_test: unseen data for final validation
    X_train, y_train, X_test, y_test = benchmark._generate_data()
 
    # The runner handles the evaluation of candidate expressions
    # It takes input features and returns predictions
    runner = SymbolicRegressionRunner(
        data_train=(X_train, y_train),
        data_val=(X_test, y_test), 
        data_test=(X_test, y_test)   
    )

 
    # Get grammar fron the benchmark
    grammar =  benchmark.grammar()

    # load configuraitons for running the project
    ge_config = FinchConfig.from_yaml('config.yaml')
    
    # The mapper converts integer arrays (genotypes) into valid expression trees (phenotypes)
    mapper = GenotypeMapper(
        grammar=grammar,
        max_wraps=ge_config.ge[Keys.MAX_WRAPS],
        max_recursion_depth=ge_config.ge[Keys.MAX_RECURSION_DEPTH],
        random_state=ge_config.experiment[Keys.RANDOM_SEED]
    )

    # Coordinates the evaluation of individuals in the population
    # Can evaluate individuals in parallel for better performance
    fitness_evaluator = FitnessEvaluator(
        runner=runner, 
        fitness_functions=RMSEFitness(),  
        mapper=mapper,
        parallel_config=ge_config.parallel
    )
 
    # Setup Grammatical Evolution , the experiment pipeline
    ge_ = GrammaticalEvolution(
        grammar=grammar,
        fitness_evaluator=fitness_evaluator 
    ) 

    result = ge_.run()
  

    print(f"\nBest fitness (RMSE): {result.all_time_best.fitness[0]:.6f}")
    print(f"Best program: {result.all_time_best.phenotype}")

    # Test on unseen data
    test_predictions = runner.run(result.all_time_best.phenotype)
    test_rmse = np.sqrt(np.mean((test_predictions['y_pred'] - y_test) ** 2))
    print(f"Test RMSE: {test_rmse:.6f}")
    