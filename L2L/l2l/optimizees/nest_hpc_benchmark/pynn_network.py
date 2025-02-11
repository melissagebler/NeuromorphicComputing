%%file ~/pynn_net.py

import time
import pyNN.spiNNaker as sim

class Pynn_Net():
    sim.setup(1.0)
   
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
    
    def build_network(self):
    #input and populations
    #!!change input rate0
        self.input_poisson = sim.Population(1, sim.SpikeSourcePoisson(rate = self.scale*10), label = 'input')
        #input_in = sim.Population(self.NI/3, sim.SpikeSourcePoisson, label = 'input_in')
        self.pop1_ex=sim.Population(self.NE, sim.IF_curr_exp())
        self.pop1_in=sim.Population(self.NI, sim.IF_curr_exp())
        #pop2=sim.Population((self.NE/3)+(self.NI/3), sim.IF_curr_exp())

    #record spikes
        self.sample_ex = self.pop1_ex.sample(self.nrec)
        self.sample_ex.record(["spikes"])
        #self.sample_in = self.pop1_in.sample(self.nrec)
        #self.sample_in.record(["spikes"])
        
        self.connect_pops()

    def connect_pops(self):
    #connect populations
        sim.Projection(self.input_poisson, self.pop1_ex, sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(weight=self.weight_excitatory, delay=self.delay))
        sim.Projection(self.input_poisson, self.pop1_in, sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(weight=self.weight_excitatory, delay=self.delay))
        sim.Projection(self.pop1_ex, self.pop1_in, sim.FixedNumberPreConnector(self.CE, with_replacement = True, allow_self_connections = False), synapse_type=sim.StaticSynapse(weight=self.weight_excitatory, delay=self.delay))
        sim.Projection(self.pop1_ex, self.pop1_ex, sim.FixedNumberPreConnector(self.CE, with_replacement = True, allow_self_connections = False), synapse_type=sim.StaticSynapse(weight=self.weight_excitatory, delay=self.delay))
        sim.Projection(self.pop1_in, self.pop1_in, sim.FixedNumberPreConnector(self.CI, with_replacement = True, allow_self_connections = False), synapse_type=sim.StaticSynapse(weight=self.weight_inhibitory, delay=self.delay))
        sim.Projection(self.pop1_in, self.pop1_ex, sim.FixedNumberPreConnector(self.CI, with_replacement = True, allow_self_connections = False), synapse_type=sim.StaticSynapse(weight=self.weight_inhibitory, delay=self.delay))

    def run_simulation(self):
        start = time.time()
        self.build_network()
        end = time.time()
        buildtime = end - start
        
        start = time.time()
        sim.run(300)
        #spikes1=self.sample_ex.get_data(["spikes"]).segments[0].spiketrains
        #spikes2=self.sample_in.get_data(["spikes"]).segments[0].spiketrains
        average_rate = self.sample_ex.mean_spike_count()
        
        temp = time.time()
        simtime = temp - start
        
        if average_rate/simtime < 100:
            #sim.reset()
            sim.run(2000)
            average_rate = self.sample_ex.mean_spike_count()
            sim.end()
            end = time.time()
            simtime = end - start
            average_rate = average_rate/simtime
        else:
            sim.end()
            end = time.time()
            simtime = end - start
            #average_rate = NaN
        
        #sim.end()
        return average_rate, buildtime, simtime

if __name__ == "__main__":    
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
    
    """spikes1=pop1.get_data(["spikes"]).segments[0].spiketrains
    v1 = pop1.get_data(["spikes","v"]).segments[0].filter(name='v')[0]

    spikes2=pop2.get_data(["spikes", "v"]).segments[0].spiketrains
    v2 = pop2.get_data(["spikes","v"]).segments[0].filter(name='v')[0]*/



    from pyNN.utility.plotting import Figure, Panel

    Figure(Panel(spikes1))
    print(v1)

    Figure(Panel(spikes2))
    print(v2)"""