import time
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
#from .pynn_network import Pynn_Net
import numpy as np
import random

import nmpi
#not needed any longer: import hbp_service_client
client = nmpi.Client()
import os 
#import time
import ebrains_drive
from ebrains_drive.client import DriveApiClient

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
                      'CE':         random.uniform(50     , 100),
                      'CI':         random.uniform(25     , 50),
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
                      'CE':         np.clip(individual['CE']       , 50, 100),
                      'CI':         np.clip(individual['CI']       , 25, 50),
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
        """net = Pynn_Net(scale=self.scale, 
                                   CE=CE, 
                                   CI=CI, 
                                   weight_excitatory=weight_ex, 
                                   weight_inhibitory=weight_in, 
                                   delay=delay,
                                   nrec=self.nrec
                                   )"""
        """net = Pynn_Net(scale=0.01, 
                                   CE=50, 
                                   CI=10, 
                                   weight_excitatory=15, 
                                   weight_inhibitory=-100, 
                                   delay=5,
                                   nrec=5
                                   )"""
        average_rate = 1 #net.run_simulation()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "pynn_network.py")
        with open(file_path, "r") as file:
            net = file.readlines()
        
        filename = "netcall" + str(time.time())
        with open(os.path.join(os.path.expanduser('~'),f'{filename}.py'), 'a') as f:
            for line in net:
                f.write(line)
            f.write('if  __name__ == "__main__":\n')
            f.write(f"    net = Pynn_Net(scale = {self.scale}, CE = {CE}, CI = {CI}, weight_excitatory = {weight_ex}, weight_inhibitory = {weight_in}, delay = {delay}, nrec = {self.nrec})\n    average_rate, buildtime, simtime = net.run_simulation()\n")
            f.write('    with open("average_rate.txt", "w") as f:\n')
            f.write('        f.write(f"{average_rate}")\n')
            f.write('    with open("times.txt", "w") as f:\n')
            f.write('        f.write(f"{buildtime}\\n{simtime}")')
        
        job = client.submit_job(source=f'~/{filename}.py',
                        platform=nmpi.SPINNAKER,
                        collab_id='nmc-test-melgeb',
                        command="run.py",
                        wait=True)
        print(job["log"])
        filenames = client.download_data(job, local_dir=os.path.expanduser("~"))
        print("Fetched the files",filenames)

        with open(filenames[0], 'r') as file:
            average_rate = float(file.read())
        with open(filenames[2], 'r') as file:
            times = file.readlines()
        buildtime = float(times[0].strip())
        simtime = float(times[1])
        print("average_rate: " + str(average_rate) + '\nBuildtime: ' + str(buildtime) + '\nSimtime: ' + str(simtime))
        
        desired_rate = 10
        fitness = -abs(average_rate - desired_rate) # TODO: is this a sensible way to calculate fitness?
        print("fitness:", fitness)
        return (fitness,) 



    
    




