import os
import numpy as np
import pandas as pd
import scipy.spatial.distance as sd
from tqdm import tqdm

def FRAP_simulation_with_binding(nparticles,
                                 D,
                                 R_bound_initial,
                                 k_bound,
                                 k_free,
                                 T, 
                                 dt, 
                                 xmax,
                                 FRAP_center, 
                                 FRAP_radius,
                                 save_particles=False,
                                 dimension=3):
    """
    Simulate FRAP experiment with binding of particles with receptors
    Args:
        nparticles (int): total number of particles
        D (float): diffusion coefficient of particles
        R_total (int): total number of receptors
        R_bound_initial (int): initial number of bounded receptors (<R_total)
        k_bound (float): binding affinity
        k_free (float): diassociation constant
        T (float): total time of simulation
        dt (float): time step
        xmax (int): size of cube for space
        dimension (int): dimension of space
        FRAP_center (array): centre of FRAP
        FRAP_radius (array): size of FRAP
        save_particles (str or False): Give folder name to save particle position in each timestep (only for 3D simulations)
    """
    particle_coords, status, binding_status = setup_initial_conditions(nparticles,        
                                                                       xmax,
                                                                       dimension,           
                                                                       FRAP_center,       
                                                                       FRAP_radius,                
                                                                       R_bound_initial)

    nt = int(T/dt)
    on_num = np.zeros(nt) 
    off_num = np.zeros(nt)

    for i in tqdm(range(nt)):
        # diffuse unbounded particles
        particle_coords = diffuse_step(particle_coords,         
                                       binding_status, 
                                       D,             
                                       dt,             
                                       nparticles,    
                                       xmax,           
                                       dimension)

        # free bounded particles if there are bounded particles
        bound_particle_indices = np.where(binding_status)[0]
        freed_particles = free_bounded_particles(bound_particle_indices,
                                                 k_free,
                                                 dt)

        # bind free particles if there are free receptors
        free_particle_indices = np.where(~binding_status)[0]
        binding_particles = bind_free_particles(free_particle_indices,
                                                k_bound,
                                                dt)
        
        binding_status = make_new_binding_status(binding_status, freed_particles, binding_particles)

        # record FRAP data
        on_num[i], off_num[i] = get_frap_data(particle_coords, status, FRAP_center, FRAP_radius)

        # save particle positions in a folder 
        if save_particles:
            if not os.path.exists(save_particles):
                os.makedirs(save_particles)
            
            # make dataframe
            x = particle_coords[:,0]
            y = particle_coords[:,1]
            z = particle_coords[:,2]
            df = pd.DataFrame({'x':x, 'y':y, 'z':z, 'status':status, 'binding':binding_status})
            df.to_csv(os.path.join(save_particles, f"time_{i}.csv"))

    return on_num, off_num

def setup_initial_conditions(nparticles,        # number of diffusible particles
                             xmax,              # maximum x, y, z
                             dimension,         # 2 or 3
                             FRAP_center,       # centre of FRAP experiment
                             FRAP_radius,       # radius of FRAP
                             R_bound_initial):  # number of bounded receptors (with particles)
    """Set up initial conditions for FRAP with binding"""
    # randomly assign initial coordinates for all particles
    initial_coords = np.random.randint(low=0, high=xmax, size=(nparticles, dimension))
    # calculate distances of the initial coord to frap centre
    in_frap = sd.cdist(initial_coords, np.array([FRAP_center]))
    # assign "status" to the particles in the system - "on" for fluorescent particles, "off" for bleached particles
    status = ["off" if in_frap[i]<FRAP_radius else "on" for i in range(nparticles)]
    # randomly choose particles to be bounded to receptors
    rand_bound = np.random.choice(nparticles, R_bound_initial, False)
    # assign binding status to particles - True for bounded particles False for free particles
    binding = np.array([True if i in rand_bound else False for i in range(nparticles)])
    return initial_coords, status, binding

def periodic_bc(position, lower_bound, upper_bound):
    """Implement boundary conditions"""
    # Replace values lower than lower_bound with upper_bound in-place
    position[position < lower_bound] = upper_bound
    # Replace values higher than upper_bound with lower_bound in-place
    position[position > upper_bound] = lower_bound
    return position

def diffuse_step(coord,          # current coordinates of particles
                 binding_status, # True particles do not move, False particle diffuse
                 D,              # Diffusion coefficient
                 dt,             # time step
                 nparticles,     # total number of particles
                 xmax,           
                 dimension):
    """Diffuse and update positions for particles that are unbounded (binding_status==False)"""
    gaussian_samples = np.sqrt(2*D*dt) * np.random.randn(nparticles, dimension)
    # convert binding status to integer
    binding_int = 1 - np.array(binding_status).astype(int)
    # convert to 2D or 3D array (according to dimension)
    cover_binding = np.column_stack((binding_int,)*dimension)
    new_coord = coord + gaussian_samples.astype(int) * cover_binding
    new_coord = periodic_bc(new_coord, 0, xmax)
    return new_coord

def free_bounded_particles(bounded_particles, k_free, dt):
    """Returns the indices of the bounded particles that are freed from receptors"""
    freed_particles = bounded_particles[np.random.rand(len(bounded_particles))<k_free*dt] 
    return freed_particles

def bind_free_particles(free_particles, k_bound, dt):
    """Returns the indices of the freed particles that are bounded from receptors"""
    binding_particles = free_particles[np.random.rand(len(free_particles))<k_bound*dt]
    return binding_particles

def make_new_binding_status(binding_status, 
                            freed_particles,
                            binding_particles):
    """Update binding status"""
    binding_status[freed_particles] = False
    binding_status[binding_particles] = True
    return binding_status

def update_receptor_number(binding_status, R_total):
    """Update number of free and bounded particles"""
    R_bound = np.sum(binding_status)
    R_free = R_total - R_bound
    return R_free, R_bound

def get_frap_data(coord, status, FRAP_center, FRAP_radius):
    """Obtain FRAP data"""
    in_frap = sd.cdist(coord, np.array([FRAP_center])) < FRAP_radius
    in_frap = in_frap.flatten()
    on = np.array(status) == "on"
    off = np.array(status) == "off"
    on_num = np.sum(in_frap & on)
    off_num = np.sum(in_frap & off)
    return on_num, off_num
 