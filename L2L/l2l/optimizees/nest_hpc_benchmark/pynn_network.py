import pyNN.spiNNaker as sim

sim.setup(1.0)

#input and populations
input = sim.Population(2, sim.SpikeSourceArray(spike_times=[[0],[1]]))
pop1=sim.Population(2,sim.IF_curr_exp())
pop2=sim.Population(2,sim.IF_curr_exp())

#record spikes
pop1.record(["spikes", "v"])
pop2.record(["spikes", "v"])

#connect populations
sim.Projection(input, pop1, sim.OneToOneConnector(), synapse_type=sim.StaticSynapse(weight=50, delay=2))
sim.Projection(pop1, pop2, sim.OneToOneConnector(), synapse_type=sim.StaticSynapse(weight=50, delay=2))

sim.run(10)

#extract data
spikes1=pop1.get_data(["spikes", "v"]).segments[0].spiketrains
v1 = pop1.get_data(["spikes","v"]).segments[0].filter(name='v')[0]

spikes2=pop2.get_data(["spikes", "v"]).segments[0].spiketrains
v2 = pop2.get_data(["spikes","v"]).segments[0].filter(name='v')[0]

sim.end

print(v1)
print(v2)