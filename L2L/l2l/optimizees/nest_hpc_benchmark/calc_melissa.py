
from network_exp import NestBenchmarkNetwork

net = NestBenchmarkNetwork(scale=10,
                                   CE=9000,#c_ex,
                                   CI=2250,#c_in,
                                   weight_excitatory=11,#w_ex,
                                   weight_inhibitory=-221.04800158270479,#w_in,
                                   delay=0.1,#delay,
                                   nrec=500
                           )

average_rate = net.run_simulation()
print('avg rate', average_rate, 'Hz')
