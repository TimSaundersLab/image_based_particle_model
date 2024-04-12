# Model for FCS experiment for free environment (no cells)
import numpy as np
import scipy.spatial.distance as sd
from tqdm import tqdm

def FCS_simulation(nparticles, 
                   D, 
                   T, 
                   dt, 
                   xmax, 
                   dimension, 
                   FCS_center, 
                   FCS_radius):
    """
    Run FCS experiment
    Args:
        nparticles (int): total number of particles in system
        D (float): true diffusion coefficient of particles
        T (float): final time of simulation
        dt (float): time step size
        xmax (int): size of square/ cube (pixels)
        dimension (int): dimension of space
        FCS_center (array): coordinates of centre of photobleaching region
        FCS_radius (array): radius of photobleaching region
    """
    coords = setup_initial_coords(nparticles, xmax, dimension)
    nt = int(T/dt)
    fcs_num = np.zeros(nt)
    for i in tqdm(range(nt)):
        coords = diffuse_step(coords, D, dt, nparticles, xmax, dimension)
        fcs_num[i] = get_fcs_data(coords, FCS_center, FCS_radius)
    return fcs_num

def setup_initial_coords(nparticles, xmax, dimension):
    """Set up initial conditions"""
    initial_coords = np.random.randint(low=0, high=xmax, size=(nparticles, dimension))
    return initial_coords

def diffuse_step(coord, D, dt, nparticles, xmax, dimension):
    """Diffuse particles"""
    gaussian_samples = np.sqrt(2*D*dt) * np.random.randn(nparticles, dimension)
    new_coord = coord + gaussian_samples.astype(int)
    new_coord = periodic_bc(new_coord, 0, xmax)
    return new_coord

def get_fcs_data(coord, FCS_center, FCS_radius):
    """Obtain FCS data"""
    in_fcs = sd.cdist(coord, np.array([FCS_center])) < FCS_radius
    in_fcs = in_fcs.flatten()
    return np.sum(in_fcs)

def periodic_bc(position, lower_bound, upper_bound):
    """Implement boundary conditions"""
    # Replace values lower than lower_bound with upper_bound in-place
    position[position < lower_bound] = upper_bound
    # Replace values higher than upper_bound with lower_bound in-place
    position[position > upper_bound] = lower_bound
    return position