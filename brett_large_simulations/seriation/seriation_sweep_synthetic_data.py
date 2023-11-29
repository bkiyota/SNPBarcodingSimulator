#!/usr/bin/env python
# coding: utf-8

import os
# https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
# If run in Pycharm, need to set those as environment variables in Pycharm directly.
num_threads = "1"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import time
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import autograd
import autograd.numpy as anp
import json
import pathos.multiprocessing as mp
import re
import sweetsweep

import seriation
# sys.path.append(".")
from gwot_sim import Simulation

outdir = "../results-seriation-synthetic/"

# Parameter dictionary
all_param_dict = {}

# setup simulation parameters
# all_param_dict["sim_path"] = ""
all_param_dict["sim_path"] = "/home/matthieu/Work/Postdoc-UBC/Projects/seriation/results-seriation-synthetic/sweep_14-21-28-04_simulations_dim-N-T"
# all_param_dict["sim_path"] = ""    # Set to "" to compute the simulation, or to an "npy" file, or to the sweep folder where to find the simulations
all_param_dict["reduce_dim"] = False
all_param_dict["rand_seed"] = 0
all_param_dict["dim"] = 20  # dimension of simulation
all_param_dict["N"] = 32  # number of particles per timepoint
all_param_dict["T"] = 16  # number of timepoints
all_param_dict["sim_steps"] = 1000  # number of steps to use for Euler-Maruyama method
all_param_dict["D"] = 1.0  # diffusivity
all_param_dict["t_final"] = 1  # simulation run on [0, t_final]
all_param_dict["max_workers"] = 16   # Set to 1 to run on a single thread and get reproducible (deterministic) results

# Marginal distance parameters
all_param_dict["marg_dist"] = "Unb-sink"  # Sink, Unb-sink, Sink-div, EMD, MMD-e, MMD-g, MMD-l
all_param_dict["comp_diag"] = False
all_param_dict["epsilon"] = 1  # entropic regularization
all_param_dict["tau"] = 0.95  # unbalanced regularization
all_param_dict["sink_iter"] = 100
all_param_dict["MMD_blur"] = 1 # blur parameter for MMDg and MMDl

# Similarity parameters
all_param_dict["similarity"] = "Exp-kernel"  # Exp-kernel, QOT
all_param_dict["sim_sigma"] = 0.5  # std dev for kernel transforming distances to similarities. Low values act as a soft threshold
all_param_dict["sim_cut"] = 0   # (hard) threshold below which we set similarity to 0.
all_param_dict["dist_power"] = 2  # Power of the distance
all_param_dict["dist_mat_norm"] = 2  # Method of normalization of the distance matrix: 1,2 or 3.
all_param_dict["qot_reg"] = 1.5  # Regularization factor for QOT

# UMAP parameters
all_param_dict["neighb"] = 5
all_param_dict["mindist"] = 1
all_param_dict["epochs"] = 1000
all_param_dict["metric"] = "precomputed"



# Swept parameters
# # all_param_dict["dim"] = [2,5,10,20,50]  # dimension of simulation
# # all_param_dict["N"] = [2,4,8,16,32,64,128,256]  # number of particles per timepoint
# # all_param_dict["N"] = [32,64]  # number of particles per timepoint
# # all_param_dict["T"] = [5,10,20,50,100,200,500,1000]  # number of timepoints
# # all_param_dict["T"] = [4,8,16,32,64,128,256]  # number of timepoints
# all_param_dict["marg_dist"] = ["Sink","Unb-sink","Sink-div","EMD","MMD-e","MMD-g","MMD-l"]  # sink, unb-sink, sink-div, EMD, MMD-e, MMD-g, MMD-l
# # all_param_dict["marg_dist"] = ["Sink","Unb-sink","Sink-div"]
# # all_param_dict["marg_dist"] = ["Unb-sink"]
# # all_param_dict["comp_diag"] = [False,True]
# # all_param_dict["comp_diag"] = [False]
# all_param_dict["sim_sigma"] = [0.9,0.7,0.5,0.3,0.1]
# # all_param_dict["epsilon"] = [0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5]  # entropic regularization
# # all_param_dict["epsilon"] = [0.01,0.1,0.2,0.5,1,2]  # entropic regularization
# all_param_dict["epsilon"] = [0.03,0.1,0.3,1]  # entropic regularization
# # all_param_dict["epsilon"] = [0.5,1,2]  # entropic regularization
# # all_param_dict["epsilon"] = [0.05]  # entropic regularization
# # all_param_dict["tau"] = [0.9,0.95,0.99,1]  # unbalanced regularization
# all_param_dict["MMD_blur"] = [0.3,1,3] # blur parameter for MMDg and MMDl

all_param_dict["similarity"] = ["Exp-kernel","QOT"]
# all_param_dict["marg_dist"] = ["Sink","Unb-sink","Sink-div","EMD","MMD-e","MMD-g","MMD-l"]  # sink, unb-sink, sink-div, EMD, MMD-e, MMD-g, MMD-l
all_param_dict["marg_dist"] = ["MMD-e","MMD-g","MMD-l"]  # sink, unb-sink, sink-div, EMD, MMD-e, MMD-g, MMD-l
all_param_dict["sim_sigma"] = [0.9,0.7,0.5,0.3,0.1]
all_param_dict["dist_power"] = [1,2,3]
# all_param_dict["qot_reg"] = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
all_param_dict["qot_reg"] = [1,1.05,1.1,1.2,1.3,1.5,1.7,2]


specific_dict = {}
if isinstance(all_param_dict["epsilon"],list):
    specific_dict["epsilon"] = {"marg_dist": ["Sink","Unb-sink","Sink-div"]}
if isinstance(all_param_dict["MMD_blur"],list):
    specific_dict["MMD_blur"] = {"marg_dist": ["MMD-g","MMD-l"]}
if isinstance(all_param_dict["sim_sigma"], list):
    specific_dict["sim_sigma"] = {"similarity": ["Exp-kernel"]}
if isinstance(all_param_dict["qot_reg"], list):
    specific_dict["qot_reg"] = {"similarity": ["QOT"]}

# skip_exps = []
# skip_exps.append({"neighb":[10], "T":[9]})


# Extract the parameters to sweep over
param_sweep = {}
for p in all_param_dict:
    if isinstance(all_param_dict[p],list):
        param_sweep[p] = all_param_dict[p]


# Main folder for the sweep
my_sweep_dir = os.path.join(outdir,"sweep_"+datetime.datetime.now().strftime('%H-%M-%S-%f')[:-4])
os.makedirs(my_sweep_dir, exist_ok=True)

# Save all parameters
json.dump(all_param_dict, open(os.path.join(my_sweep_dir, "all_parameters.json"), "w"), indent=0, sort_keys=True)

# Name of the csv file to save (one row per experiment)
csv_filename = "results.csv"

# Names of image results, to put them in the sweep file
img_result_dict = {}
img_result_names = ["input-data","dist-matrix","dist-hist","simil-hist","simil-vs-dist","all","fvec_lap",
                    "fvec_norm_lap_DLD","fvec_norm_lap_DL","umap","umap2d","lengths"]
for n in img_result_names:
    img_result_dict[n] = n + ".png"

# Save the param_sweep file
params = param_sweep.copy()
# Add parameters for the viewer if you need (see README.md)
params["viewer_filePattern"] = list(img_result_dict.values())
# params["viewer_cropLBRT"] = [0, 0, 0, 0]
params["viewer_resultsCSV"] = csv_filename
json.dump(params, open(os.path.join(my_sweep_dir, "sweep.txt"), "w"))


def simulate_input_data(param_dict):

    T = param_dict["T"]
    N = param_dict["N"]
    D = param_dict["D"]
    dim = param_dict["dim"]
    sim_steps = param_dict["sim_steps"]
    t_final = param_dict["t_final"]
    max_workers = param_dict["max_workers"]

    # TODO: Maybe do a potential that attracts in all dimensions, not just the first 2?
    # setup potential function
    def Psi(x, t, dim=dim):
        x0 = np.array([1, 1] + [0, ] * (dim - 2))
        x1 = -np.array([1, 1] + [0, ] * (dim - 2))
        force = 1.25
        # x0 = np.ones(dim)
        # x1 = -np.ones(dim)
        # force = 0.3
        return force * anp.sum((x - x0) * (x - x0), axis=-1) * anp.sum((x - x1) * (x - x1), axis=-1) + 10 * anp.sum(
            x[:, 2:] * x[:, 2:], axis=-1)

    # get gradient
    dPsi = autograd.elementwise_grad(Psi)

    print("\nN =", N, "samples per timepoint.")
    print("dim =", param_dict["dim"])

    # function for particle initialisation
    ic_func = lambda N, d: np.random.randn(N, d) * 0.1
    # def ic_func(N, d):
    #     return np.random.randn(N, d)*0.1

    # setup simulation object
    pool = mp.ProcessingPool(max_workers) if max_workers > 1 else None

    sim = Simulation(V=Psi, dV=dPsi, birth_death=False, N=np.repeat(N, T), T=T, d=dim, D=D,
                     t_final=t_final, ic_func=ic_func, pool=pool)

    print("Simulating dataset")
    # sample from simulation
    sim.sample(steps_scale=int(sim_steps / sim.T), trunc=N)

    # Save samples
    X = sim.x.reshape(T, N, -1)

    return X


def find_simulation_data(param_dict):
    # Get all directories in the simulation directory
    alldirs = [os.path.basename(f) for f in os.scandir(param_dict["sim_path"]) if f.is_dir()]
    # Get the list of parameters that vary in that simulation dir
    sim_sweep_dict = json.load(open(os.path.join(param_dict["sim_path"], "sweep.txt"), 'r'))
    # Remove the viewer parameters
    sim_sweep_params = [k for k in sim_sweep_dict.keys() if not k.startswith("viewer")]
    # Remove all dirs that don't match all parameters.
    for param in sim_sweep_params:
        alldirs = [d for d in alldirs if re.search(param + str(param_dict[param]) + "(_|$)", d)]
    # There should be only one left.
    if len(alldirs) > 1:
        print("WARNING: The parameter set matches multiples simulation directories. Taking the first one:", alldirs[0])
    if len(alldirs) == 0:
        print("WARNING: The parameter set doesn't match any simulation directory. Skipping this experiment.")
        return
    # Load the simulation data
    X = np.load(os.path.join(param_dict["sim_path"], alldirs[0], "input-data.npy"))
    return X


def my_experiment(exp_id, exp_param_dict, exp_dir):

    # Combine varying and non-varying parameters
    param_dict = all_param_dict.copy()
    param_dict.update(exp_param_dict)

    print("\nExperiment #%d:"%exp_id, param_dict)
    t0 = time.time()

    # Set random seed
    rng = np.random.default_rng(param_dict["rand_seed"])

    # Load or simulate input data
    if param_dict["sim_path"]:
        if param_dict["sim_path"].endswith(".npy"):
            X = np.load(param_dict["sim_path"])
        else:
            X = find_simulation_data(param_dict)
    else:
        X = simulate_input_data(param_dict)
        np.save(os.path.join(exp_dir, "input-data"), X)

    # Plot samples
    T = param_dict["T"]
    N = param_dict["N"]
    t_final = param_dict["t_final"]
    plt.figure()
    plt.scatter(np.linspace(0, t_final, T).repeat(N), X[:,:,0].flatten(), alpha=0.25, color="red")
    plt.xlabel("t")
    plt.ylabel("dim 1")
    plt.ylim(-1.5,1.5)
    plt.title("Input data")
    plt.savefig(os.path.join(exp_dir,img_result_dict["input-data"]))
    plt.close()

    # # Code to only generate the simulations
    # experiment_time = time.time()-t0
    # print("Time:",experiment_time,"\n")
    # return {"time": "%0.2g"%experiment_time}

    # Compute distance matrix
    dist_mat, result_dict1 = seriation.compute_distance_matrix(param_dict,X)

    # Perform seriation, save and plot results
    result_dict2 = seriation.seriation_save_and_plot(exp_dir, param_dict, dist_mat, X, img_result_dict)

    # Scalar results
    experiment_time = time.time() - t0
    print("Time:",experiment_time,"\n")

    # Return additional scalar results as a dictionary, where keys will be the corresponding columns in the CSV.
    # Return the results either as their original dtype, or as how you want them to appear in the viewer:
    return {"time": "%0.2g"%experiment_time,
            **result_dict1,
            **result_dict2
            }


# Run the sweep, the function creates and fills the CSV for you
sweetsweep.parameter_sweep(param_sweep, my_experiment, my_sweep_dir, result_csv_filename=csv_filename, start_index=0, specific_dict=specific_dict)
# sweep.parameter_sweep_parallel(param_sweep, my_experiment, my_sweep_dir, result_csv_filename=csv_filename)
