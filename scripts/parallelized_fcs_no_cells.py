# Example for running samples of FCS in a 3D cube without cells
# Parallelized to reduce computational time
import import_helper
import_helper.add_models()
import os
import numpy as np
import pandas as pd
import particle_models.fcs_no_cells as fcs
from joblib import Parallel, delayed
from tqdm import tqdm

FOLDER = "../data/fcs_no_cell_R5"
dimension = 3
diff_coefs = [2, 5, 10, 20, 30, 50, 80, 100]
n_sample = 100
T = 600
dt = 0.1

particle_number = 200000
max_x = 1400
FCS_midpoint = np.array([700, 700, 700])
FCS_radius = 5

def run_many_FCS_experiment(nsamples, diffusion_coef, subfolder_path):
    # make directory to store data
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    for i in range(nsamples):
        fcs_num = fcs.FCS_simulation(particle_number,
                                     diffusion_coef, 
                                     T, 
                                     dt, 
                                     max_x, 
                                     dimension, 
                                     FCS_midpoint, 
                                     FCS_radius)
        # save frap data as dataframes
        df = pd.DataFrame(data={"fcs_region":fcs_num})
        df.to_csv(f"{subfolder_path}/sample_{i}.csv")

process = (delayed(run_many_FCS_experiment)(n_sample, diff_coefs[i], f"{FOLDER}/D{diff_coefs[i]}") for i in tqdm(range(len(diff_coefs))))
parallel = Parallel(n_jobs=5)(process)