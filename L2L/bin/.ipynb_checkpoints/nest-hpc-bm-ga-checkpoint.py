from l2l.utils.experiment import Experiment
import numpy as np

from l2l.optimizees.nest_hpc_benchmark import HPCBMOptimizee, HPCBMOptimizeeParameters
from l2l.optimizers.evolution import GeneticAlgorithmParameters, GeneticAlgorithmOptimizer
from l2l.optimizers.crossentropy import CrossEntropyParameters, CrossEntropyOptimizer
from l2l.optimizers.crossentropy.distribution import NoisyGaussian
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.gradientdescent.optimizer import GradientDescentOptimizer
from l2l.optimizers.gradientdescent.optimizer import RMSPropParameters


def run_experiment():
    experiment = Experiment(
        root_dir_path='../results')
    #"srun --ntasks=1 --cpus-per-task=32 --threads-per-core=1   --exact
    jube_params = { "exec": "srun --ntasks=1 --cpus-per-task=32 --threads-per-core=1 --exact python"} 
    traj, _ = experiment.prepare_experiment(
        jube_parameter=jube_params, name=f"HPCBenchmark_CrossEntropy")
        
    optimizer_choice = "ce"

    
    # nest HPC benchmark Optimizee
    optimizee_parameters = HPCBMOptimizeeParameters(scale=0.05,
                                                    nrec=100
                                                    )
    optimizee = HPCBMOptimizee(traj, optimizee_parameters)

    match optimizer_choice:
    ## Genetic Algorithm Optimizer
        case "ga":
            optimizer_parameters = GeneticAlgorithmParameters(seed=1580211, 
                                                              pop_size=4,
                                                              cx_prob=0.7,
                                                              mut_prob=0.7,
                                                              n_iteration=10,
                                                              ind_prob=0.45,
                                                              tourn_size=4,
                                                              mate_par=0.5,
                                                              mut_par=1)
            optimizer = GeneticAlgorithmOptimizer(traj, 
                                                  optimizee_create_individual=optimizee.create_individual,
                                                  optimizee_fitness_weights=(1,),
                                                  parameters=optimizer_parameters,
                                                  optimizee_bounding_func=optimizee.bounding_func)


    ## Cross Entropy Optimizer
        case "ce":
            optimizer_seed = 1234
            optimizer_parameters = CrossEntropyParameters(pop_size=4, rho=0.9, smoothing=0.0, temp_decay=0, n_iteration=5,
                                                         distribution=NoisyGaussian(noise_magnitude=1., noise_decay=0.99),
                                                          stop_criterion=np.inf, seed=optimizer_seed)


            optimizer = CrossEntropyOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                              optimizee_fitness_weights=(1.,),
                                              parameters=optimizer_parameters,
                                              optimizee_bounding_func=optimizee.bounding_func)

    ##Gradient Descent Optimizer
        case "gd":
            optimizer_parameters = RMSPropParameters(learning_rate=0.01, exploration_step_size=0.01,
                                           n_random_steps=5, momentum_decay=0.5,
                                           n_iteration=10, stop_criterion=np.Inf, seed=99)

            optimizer = GradientDescentOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                                 optimizee_fitness_weights=(0.1,),
                                                 parameters=optimizer_parameters,
                                                 optimizee_bounding_func=optimizee.bounding_func)
    ##None of the above
        case _:
            print("No valid optimizer chosen")

    
    
    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=optimizer_parameters,
                              optimizee_parameters=optimizee_parameters)
    # End experiment
    experiment.end_experiment(optimizer)


def main():
    run_experiment()



if __name__ == '__main__':
    main()
