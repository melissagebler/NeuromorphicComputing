from network import NestBenchmarkNetwork
import random

scale = 0.05
nrec = 100

individual = {  'weight_ex':  random.uniform(0     , 200),
                'weight_in':  random.uniform(-1000  , 0),
                'CE':         500, #random.uniform(0     , 1),
                'CI':         100, #random.uniform(0     , 1),
                'delay':      random.uniform(0.1   , 10),
                'nrec':       100
                }   


weight_ex = individual['weight_ex']
weight_in = individual['weight_in']
CE = individual['CE']
CI = individual['CI']
delay = individual['delay']

net = NestBenchmarkNetwork(scale, CE, CI, weight_ex, weight_in, delay, nrec)
average_rate = net.run_simulation()

print("")
print("average firing rate:", average_rate)