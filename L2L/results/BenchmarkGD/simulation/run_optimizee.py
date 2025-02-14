import pickle
import sys
import gc
import os
import logging
import socket
import time
worker_id = sys.argv[1]
logfilename = f"/mnt/user/shared/NMC test collab for user melgeb/SpiNNaker/NeuromorphicComputing/L2L/results/BenchmarkGD/simulation/individual_logs/workers_{worker_id}.wlog"
logging.basicConfig(filename=logfilename, filemode="a", level=logging.INFO)
logger = logging.getLogger("Optimizee")
logger.info(socket.gethostname())
outputpipename = f"/mnt/user/shared/NMC test collab for user melgeb/SpiNNaker/NeuromorphicComputing/L2L/results/BenchmarkGD/simulation/individual_logs/outputpipe_{worker_id}"
outputpipe = open(outputpipename, "wb")
inputpipename = f"/mnt/user/shared/NMC test collab for user melgeb/SpiNNaker/NeuromorphicComputing/L2L/results/BenchmarkGD/simulation/individual_logs/inputpipe_{worker_id}"
inputpipe = open(inputpipename, "r")
running = 1
while running:
    try:
        logger.info(f"Receiving")
        params = ""
        while not params:
            params = inputpipe.readline()
            time.sleep(5)
        logger.info(f"Params received: {params}")
        params = params.split()
        logger.info(params)
        generation = params[0]
        idx = params[1]
        running = int(params[2])
        if not running:
            break
        handle_trajectory = open("/mnt/user/shared/NMC test collab for user melgeb/SpiNNaker/NeuromorphicComputing/L2L/results/BenchmarkGD/simulation/trajectories/op_trajectory_"+ str(generation) + ".bin", "rb")
        trajectory = pickle.load(handle_trajectory)
        handle_trajectory.close()
        handle_optimizee = open("/mnt/user/shared/NMC test collab for user melgeb/SpiNNaker/NeuromorphicComputing/L2L/results/BenchmarkGD/simulation/optimizee.bin", "rb")
        optimizee = pickle.load(handle_optimizee)
        handle_optimizee.close()

        logger.info("Trajectory access")
        logger.info(trajectory.individuals)
        logger.info(trajectory.retry)
        logger.info(len(trajectory.individuals[int(generation)]))
        trajectory.individual = trajectory.individuals[int(generation)][int(idx)] 
        res = optimizee.simulate(trajectory)

        handle_res = open("/mnt/user/shared/NMC test collab for user melgeb/SpiNNaker/NeuromorphicComputing/L2L/results/BenchmarkGD/simulation/results/results_"+ str(generation) + "_" + str(idx) + ".bin", "wb")
        pickle.dump(res, handle_res, pickle.HIGHEST_PROTOCOL)
        handle_res.close()
        del optimizee
        outputpipe.write(f"0\n".encode('ascii'))
        outputpipe.flush()
        logger.info(f"Finished {idx}")
        gc.collect()
    except Exception as e:
        logger.info(str(e))
        logger.info(f"Params received in except: {params}")
        if params == "":
            continue
        #else:
        #    sys.stderr.write(b"1")
        #    sys.stderr.flush()
outputpipe.close()
inputpipe.close()