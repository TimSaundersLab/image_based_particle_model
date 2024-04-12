# Model for FRAP experiment for free environment (no cells)
import os
import numpy as np
import pandas as pd
import scipy.spatial.distance as sd
from tqdm import tqdm

def FRAP_simulation(nparticles, 
                    D, 
                    T, 
                    dt, 
                    xmax, 
                    dimension, 
                    FRAP_center, 
                    FRAP_radius,
                    save_particles=False):
    """
    Run FRAP experiment
    Args:
        nparticles (int): total number of particles in system
        D (float): true diffusion coefficient of particles
        T (float): final time of simulation
        dt (float): time step size
        xmax (int): size of square/ cube (pixels)
        dimension (int): dimension of space
        FRAP_center (array): coordinates of centre of photobleaching region
        FRAP_radius (array): radius of photobleaching region
        save_particles (str or False): Give folder name to save particle position in each timestep (only for 3D simulations)
    """
    
    coords, status = setup_initial_coords(nparticles, xmax, dimension, FRAP_center, FRAP_radius)
    nt = int(T/dt)
    on_num = np.zeros(nt)
    off_num = np.zeros(nt)

    for i in tqdm(range(nt)):

        if save_particles:
            # only for 3D simulations
            if not os.path.exists(save_particles):
                os.makedirs(save_particles)
            
            # make dataframe
            x = coords[:,0]
            y = coords[:,1]
            z = coords[:,2]
            df = pd.DataFrame({'x':x, 'y':y, 'z':z, 'status':status})
            df.to_csv(os.path.join(save_particles, f"time_{i}.csv"))

        coords = diffuse_step(coords, D, dt, nparticles, xmax, dimension)
        on_num[i], off_num[i] = get_frap_data(coords, status, FRAP_center, FRAP_radius)

    return on_num, off_num

def setup_initial_coords(nparticles, xmax, dimension, FRAP_center, FRAP_radius):
    """
    To set up initial conditions for FRAP experiments
    Args:
        nparticles (int): total number of particles in system
        xmax (int): size of square/ cube (pixels)
        dimension (int): dimension of space 
        FRAP_center (array): coordinates of centre of photobleaching region
        FRAP_radius (array): radius of photobleaching region
    """
    initial_coords = np.random.randint(low=0, high=xmax, size=(nparticles, dimension))
    in_frap = sd.cdist(initial_coords, np.array([FRAP_center]))
    status = ["off" if in_frap[i]<FRAP_radius else "on" for i in range(nparticles)]
    return initial_coords, status

def diffuse_step(coord, D, dt, nparticles, xmax, dimension):
    """Diffuse particles"""
    gaussian_samples = np.sqrt(2*D*dt) * np.random.randn(nparticles, dimension)
    new_coord = coord + gaussian_samples.astype(int)
    new_coord = periodic_bc(new_coord, 0, xmax)
    return new_coord

def get_frap_data(coord, status, FRAP_center, FRAP_radius):
    """Returns FRAP data"""
    in_frap = sd.cdist(coord, np.array([FRAP_center])) < FRAP_radius
    in_frap = in_frap.flatten()
    on = np.array(status) == "on"
    off = np.array(status) == "off"
    on_num = np.sum(in_frap & on)
    off_num = np.sum(in_frap & off)
    return on_num, off_num

# periodic or reflexive boundary should be the same for cubic empty environment
def periodic_bc(position, lower_bound, upper_bound):
    """Implement boundary conditions"""
    # Replace values lower than lower_bound with upper_bound in-place
    position[position < lower_bound] = upper_bound
    # Replace values higher than upper_bound with lower_bound in-place
    position[position > upper_bound] = lower_bound
    return position