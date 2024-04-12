# Example for running samples of FRAP in a 3D cube without cells
# Parallelized to reduce computational time
import import_helper
import_helper.add_models()
import os
import numpy as np
import pandas as pd
import particle_models.frap_no_cells as frap
from joblib import Parallel, delayed
from tqdm import tqdm

FOLDER = "../data/frap_R300/no_cells"
dimension = 3
diff_coefs = [10, 20, 30, 50, 80, 100]
n_sample = 10
T = 2000
dt = 1

particle_number = 20000
max_x = 1400 # same as number of pixels for zebrafish images
FRAP_midpoint = np.array([700, 700, 700])
FRAP_radius = 300

def run_many_FRAP_experiment(nsamples, diffusion_coef, subfolder_path):
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    for i in range(nsamples):
        on_num, off_num = frap.FRAP_simulation(particle_number, 
                                               diffusion_coef, 
                                               T, 
                                               dt, 
                                               max_x, 
                                               dimension, 
                                               FRAP_midpoint, 
                                               FRAP_radius,
                                               False)
        df = pd.DataFrame({'on_num':on_num, 'off_num':off_num})
        df.to_csv(f"{subfolder_path}/sample_{i}.csv")

process = (delayed(run_many_FRAP_experiment)(n_sample, diff_coefs[i], f"{FOLDER}/D{diff_coefs[i]}") for i in tqdm(range(len(diff_coefs))))
parallel = Parallel(n_jobs=8)(process)