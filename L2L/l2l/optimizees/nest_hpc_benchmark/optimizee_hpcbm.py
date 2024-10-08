import time
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from .network_exp import NestBenchmarkNetwork
import numpy as np
import random

HPCBMOptimizeeParameters = namedtuple(
    'HPCBMOptimizeeParameters', ['scale', 'nrec']) # TODO: add pre-sim-time, sim-time, dt? as parameters

class HPCBMOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        self.scale = parameters.scale
        self.nrec = parameters.nrec


    def create_individual(self):
        """
        Creates and returns a random individual
        """

        individual = {'weight_ex':  random.uniform(0     , 200),
                      'weight_in':  random.uniform(-1000  , 0),
                      'CE':        500, #random.uniform(1     , 1),
                      'CI':        100, #random.uniform(1     , 1),
                      'delay':      random.uniform(0.1   , 10),
                      }   

        print("random individual:", individual) 
        
        return individual
    

    def bounding_func(self, individual):
        """
        """
        # TODO what are reasonable bounds?
        # weight_ex         originally: JE_pA = 10.77                      now range: [1, 20]?   better [0, 200]
        # weight_in         originally: g*JE_pA = -5*10.77 = -53.85        now range: [-100, -5]? better [-1000, 0]
        # CE                originally: 9000 fixed                         now: pairwise bernoulli range: [0, 1]
        # CI                originally: 2250 fixed                         now: pairwise bernoulli range: [0, 1]
        # delay             originally: 1.5                                now range: [0.1, 10]

        individual = {'weight_ex':  np.clip(individual['weight_ex'] , 0     , 200),
                      'weight_in':  np.clip(individual['weight_in'] , -1000  , -0),
                      'CE':        np.clip(individual['CE']       , 500, 500),
                      'CI':        np.clip(individual['CI']       , 100, 100),
                      'delay':      np.clip(individual['delay']     , 0.1   , 10),
                      }    
        return individual
    


    def simulate(self, traj):
        """
        """
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        weight_ex = traj.individual.weight_ex
        weight_in = traj.individual.weight_in

        CE = int(traj.individual.CE)
        CI = int(traj.individual.CI)
        delay = traj.individual.delay
        net = NestBenchmarkNetwork(scale=self.scale, 
                                   CE=CE, 
                                   CI=CI, 
                                   weight_excitatory=weight_ex, 
                                   weight_inhibitory=weight_in, 
                                   delay=delay,
                                   nrec=self.nrec
                                   )
        average_rate = net.run_simulation()

        desired_rate = 50
        fitness = -abs(average_rate - desired_rate) # TODO: is this a sensible way to calculate fitness?
        print("fitness:", fitness)
        return (fitness,) 
    

    
    
    




