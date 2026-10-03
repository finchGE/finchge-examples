
import numpy as np
from finchge.benchmarks.regression import KeijzerBenchmark
from finchge.fitness.fitness_functions import RMSEFitness
from finchge.runners.sr import SymbolicRegressionRunner
from finchge.grammar import Grammar
from finchge.grammar.mapper import GenotypeMapper
from finchge.fitness import FitnessEvaluator
from finchge.core import GrammaticalEvolution
from finchge.config import FinchConfig, Keys

 
if __name__ == '__main__':
    # Load benchmark data and grammar
    benchmark = KeijzerBenchmark(
        version=7,
        random_state=42,
        train_samples=20,
        test_samples=1000
    )
    X_train, y_train, X_test, y_test = benchmark._generate_data()
    grammar =  benchmark.grammar()


    # Prepare runner for evaluation expressions for predictions
    # Runners provide task-specific outputs as predictions to be evaluated by fitness function
    runner = SymbolicRegressionRunner(
        data_train=(X_train, y_train),
        data_val=(X_test, y_test), 
        data_test=(X_test, y_test)   
    )
    
    ge_config = FinchConfig.from_yaml('config.yaml')
    
    mapper = GenotypeMapper(
        grammar=grammar,
        max_wraps=ge_config.ge[Keys.MAX_WRAPS],
        max_recursion_depth=ge_config.ge[Keys.MAX_RECURSION_DEPTH],
        random_state=ge_config.experiment[Keys.RANDOM_SEED]
    )

    fitness_evaluator = FitnessEvaluator(
        runner=runner,
        fitness_functions=RMSEFitness(),
        mapper=mapper,
        parallel_config=ge_config.parallel
    )

 
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
    