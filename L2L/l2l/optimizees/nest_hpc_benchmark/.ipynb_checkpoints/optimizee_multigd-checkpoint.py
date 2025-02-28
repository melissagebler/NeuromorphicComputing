from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee

import time
import numpy as np
import random

import nmpi
#not needed any longer: import hbp_service_client
client = nmpi.Client()
import os 
#import time
import ebrains_drive
from ebrains_drive.client import DriveApiClient

MultiOptimizeeParameters = namedtuple('MultiOptimizeeParameters', ['scale','nrec'])

class MultiOptimizee(Optimizee):
    """
    This is the base class for the Optimizees, i.e. the inner loop algorithms. Often, these are the implementations that
    interact with the environment. Given a set of parameters, it runs the simulation and returns the fitness achieved
    with those parameters.
    """

    def __init__(self, traj, parameters):
        """
        This is the base class init function. Any implementation must in this class add a parameter add its parameters
        to this trajectory under the parameter group 'individual' which is created here in the base class. It is
        especially necessary to add all explored parameters (i.e. parameters that are returned via create_individual) to
        the trajectory.
        """
        super().__init__(traj)
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        
        self.scale = parameters.scale
        self.nrec = parameters.nrec

    def create_individual(self):
        """
        Create one individual i.e. one instance of parameters. This instance must be a dictionary with dot-separated
        parameter names as keys and parameter values as values. This is used by the optimizers via the
        function create_individual() to initialize the individual/parameters. After that, the change in parameters is
        model specific e.g. In simulated annealing, it is perturbed on specific criteria

        :return dict: A dictionary containing the names of the parameters and their values
        """
        individual = {'weight_ex':  random.uniform(0     , 200),
                      'weight_in':  random.uniform(-1000  , 0),
                      'CE':         random.uniform(50     , 100),
                      'CI':         random.uniform(25     , 50),
                      'delay':      random.uniform(0.1   , 10),
                      }   

        print("random individual:", individual)
        return individual

    def simulate(self, traj):
        """
        This is the primary function that does the simulation for the given parameter given (within :obj:`traj`)

        :param  ~l2l.utils.trajectory.Trajectory traj: The trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        :return: a :class:`tuple` containing the fitness values of the current run. The :class:`tuple` allows a
            multi-dimensional fitness function.

        """
        
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        print(traj.individual, flush=True) # {'eins': [1,1,1,1], ...}
        
        file_path = os.path.join(script_dir, "pynn_network.py")
        with open(file_path, "r") as file:
            net = file.readlines()
        job = []
        
        #create jobs to send off simultaneously
        for i in range(len(traj.individual['weight_ex'])):
            weight_ex = traj.individual['weight_ex'][i]
            weight_in = traj.individual['weight_in'][i]

            CE = int(traj.individual['CE'][i])
            CI = int(traj.individual['CI'][i])
            delay = traj.individual['delay'][i]
       
        
        
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

            job[i] = client.submit_job(source=f'~/{filename}.py',
                            platform=nmpi.SPINNAKER,
                            collab_id='nmc-test-melgeb',
                            command="run.py",
                            wait=False)
        
        for i in job:
            while client.job_status(job[i]) != 'finished' and client.job_status(job[i]) != 'error':
                continue
            logs = client.get_job(job, with_log=True)
            print(logs["log"])
            filenames = client.download_data(logs, local_dir=os.path.expanduser("~"))
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
            result[i] = fitness
                       
        print(result, flush=True)
        return result

        #print(traj.individual, flush=True) # {'eins': [1,1,1,1], ...}
        #res = [i/17 for i in range(len(traj.individual['eins']))]
        

    def bounding_func(self, individual):
        """
        placeholder
        """
        individual = {'weight_ex':  np.clip(individual['weight_ex'] , 0     , 200),
                      'weight_in':  np.clip(individual['weight_in'] , -1000  , -0),
                      'CE':         np.clip(individual['CE']       , 50, 100),
                      'CI':         np.clip(individual['CI']       , 25, 50),
                      'delay':      np.clip(individual['delay']     , 0.1   , 10),
                      }
        return individual