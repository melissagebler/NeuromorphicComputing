import time
import pyNN.spiNNaker as sim

class Pynn_Net():   
    def __init__(self, scale, CE, CI, weight_excitatory, weight_inhibitory, delay, nrec, extra_kernel_params=None):
        self.scale = scale
        self.NE = int(9000 * scale)
        self.NI = int(2250 * scale)
        self.CE = int(CE)
        self.CI = int(CI)

        self.weight_excitatory = weight_excitatory
        self.weight_inhibitory = weight_inhibitory

        self.delay = delay

        self.nrec = min(nrec, self.NE)
        
        self.poisson_rate = 50
        self.presimtime = 300
        self.simtime = 2000
    
    def build_network(self):
    #input and populations
    #!!change input rate0
        self.input_poisson_ex = sim.Population(self.NE, sim.SpikeSourcePoisson(rate = self.poisson_rate), label = 'input_ex')
        self.input_poisson_in = sim.Population(self.NI, sim.SpikeSourcePoisson(rate = self.poisson_rate), label = 'input_in')
        self.pop1_ex=sim.Population(self.NE, sim.IF_curr_exp())
        self.pop1_in=sim.Population(self.NI, sim.IF_curr_exp())

    #record spikes
        self.sample_ex = self.pop1_ex.sample(self.nrec)
        self.sample_ex.record(["spikes"])
        #self.sample_in = self.pop1_in.sample(self.nrec)
        #self.sample_in.record(["spikes"])
        
        self.connect_pops()

    def connect_pops(self):
    #connect populations
        sim.Projection(self.input_poisson_ex, self.pop1_ex, sim.OneToOneConnector(), synapse_type=sim.StaticSynapse(weight=self.weight_excitatory, delay=self.delay))
        sim.Projection(self.input_poisson_in, self.pop1_in, sim.OneToOneConnector(), synapse_type=sim.StaticSynapse(weight=self.weight_excitatory, delay=self.delay))
        sim.Projection(self.pop1_ex, self.pop1_in, sim.FixedNumberPreConnector(self.CE, with_replacement = True, allow_self_connections = False), synapse_type=sim.StaticSynapse(weight=self.weight_excitatory, delay=self.delay))
        sim.Projection(self.pop1_ex, self.pop1_ex, sim.FixedNumberPreConnector(self.CE, with_replacement = True, allow_self_connections = False), synapse_type=sim.StaticSynapse(weight=self.weight_excitatory, delay=self.delay))
        sim.Projection(self.pop1_in, self.pop1_in, sim.FixedNumberPreConnector(self.CI, with_replacement = True, allow_self_connections = False), synapse_type=sim.StaticSynapse(weight=self.weight_inhibitory, delay=self.delay))
        sim.Projection(self.pop1_in, self.pop1_ex, sim.FixedNumberPreConnector(self.CI, with_replacement = True, allow_self_connections = False), synapse_type=sim.StaticSynapse(weight=self.weight_inhibitory, delay=self.delay))

    def get_avg_rate(self, pop, simtime):
        """
        calculates the average firing rate (in Hz) of neurons in population pop during simulation time simtime (given in ms)
        """
        return 1000*pop.mean_spike_count()/simtime
        
    def run_simulation(self):
        sim.setup(1.0)
        
        # build network
        start = time.time()
        self.build_network()
        buildtime = time.time() - start
        
        # run presimulation
        start = time.time()
        sim.run(self.presimtime)
        average_rate = self.get_avg_rate(self.sample_ex, self.presimtime)
        
        # check if rate is plausible
        if average_rate < 100:
            # run full simulation
            sim.run(self.simtime)
            average_rate = self.get_avg_rate(self.sample_ex, self.simtime)
        
        # finish simulation for both cases
        sim.end()
        runtime = time.time() - start

        return average_rate, buildtime, runtime

"""if __name__ == "__main__":    
#extract data

    net = Pynn_Net(scale=0.01, 
                                   CE=50, 
                                   CI=10, 
                                   weight_excitatory=15, 
                                   weight_inhibitory=-100, 
                                   delay=10,
                                   nrec=5
                                   )
    average_rate, buildtime, simtime = net.run_simulation()
    with open('average_rate.txt', 'w') as f:
        f.write(f"{average_rate}")
    with open('times.txt', 'w') as f:
        f.write(f"{buildtime}\n{simtime}")    
    #spikes1 = net.run_simulation()
"""    
