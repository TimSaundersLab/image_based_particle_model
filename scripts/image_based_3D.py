# 3D image_based particle model on toy data
# Simulate for many samples and various diffusion coefficients
import import_helper
import_helper.add_models()
import particle_models.utils as u
import particle_models.image_particle_model_3D as model

import os
from tempfile import mkdtemp
import numpy as np
from tqdm import tqdm
import gc

folder_path = "../toy_data/toy_OT" # folder containing z-stacks
temp_array_filename = os.path.join(mkdtemp(), 'output_3d_array.npy')
image_array = u.make_3D_image(folder_path, temp_array_filename, dtype=np.uint8) # create array for 3D image
_,_,bound = np.shape(image_array)

temp_ind_filename = os.path.join(mkdtemp(), '3d_indices.npy') 
free_indices = u.get_3D_indices(image_array, temp_ind_filename, dtype=np.int32) # store only ECS coordinates
del image_array

bin_image = model.Binary_image(free_indices,  # creates KD-Tree
                               lower_bound=0,
                               upper_bound=bound)
del free_indices

# choose FRAP parameters
FRAP_midpoint = np.array([250, 250, 250])
FRAP_radius = 30

# parameters for particles
particle_number = 500
diffusion_coefficients =  np.array([10, 20, 30])
 
# time of iteration and samples
ns = 1
T = 100

FOLDER = "../data/image_frap/"

def run_FRAP_experiment(D, j, filename, folder=FOLDER):

    print(f"Start a FRAP experiment for D={D}")
    SUBFOLDER_PATH = os.path.join(folder, f"D{D}")
    if not os.path.exists(SUBFOLDER_PATH):
        os.makedirs(SUBFOLDER_PATH)

    time_step = 10/D # to ensure step sizes are consistent
    iteration = int(T/time_step)
    frap_model = model.FRAP_Model(N=particle_number, 
                                  binary_image=bin_image,
                                  dt=time_step,
                                  D=D,
                                  frap_center=FRAP_midpoint,
                                  frap_R=FRAP_radius,
                                  save_particles=True) # here we save particle positions
    
    # run diffusion model
    for iter in tqdm(range(iteration)):
        saving = round(float(iter*time_step), 3).is_integer() # only collect data for every integer timestep
        frap_model.step(saving)

    # save FRAP data
    num_frap = frap_model.datacollector.get_model_vars_dataframe()
    num_frap.to_csv(os.path.join(SUBFOLDER_PATH, f"{filename}.csv"))
    del num_frap
    
    # save particle positions
    particle_pos = frap_model.datacollector.get_agent_vars_dataframe()
    status = [agent.status for agent in frap_model.schedule.agents]
    u.position_to_folder(particle_pos,
                        folder_path=os.path.join(SUBFOLDER_PATH, f"{filename}_pos"),
                        dimension=3,
                        status_list=status)
    
    del particle_pos
    del frap_model
    gc.collect()

for i in range(len(diffusion_coefficients)):
    for j in range(ns):
        D = diffusion_coefficients[i]
        run_FRAP_experiment(D, j, filename=f"sample_{j}")