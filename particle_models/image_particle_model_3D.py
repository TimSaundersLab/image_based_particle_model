# Image-based particle models to simulate FRAP and FCS experiments
import numpy as np
from scipy.spatial import cKDTree
import mesa

def count_total_in_fcs(model):
    """Count number of particles in FCS region"""
    agents_in_fcs = [agent for agent in model.schedule.agents if np.linalg.norm(agent.position - model.fcs_center)<model.fcs_R]
    return len(agents_in_fcs)

def compute_off_on(model):
    """Count the number of fluorescent and non-fluorescent particles in FRAP region"""
    # if particle has dist(position, frap_center)<frap_R, count number of on and off
    agents_in_frap = [agent for agent in model.schedule.agents if np.linalg.norm(agent.position - model.frap_center)<model.frap_R]
    off_agents = [agent for agent in agents_in_frap if agent.status=="off"]
    on_agents = [agent for agent in agents_in_frap if agent.status=="on"]
    return np.array([len(off_agents), len(on_agents)])

class Particle(mesa.Agent):
    """
    An agent representing a diffusive particle
    Args:
        unique_id (int): unique ID for particle
        model (mesa.Model)
        position (array): array of particle coordinate
        status (string): fluorescent status of particle
        ("on" for fluorescent particles, "off" for bleached particles)
    """
    def __init__(self,
                 unique_id,
                 model,
                 position,
                 diffusion_coef,
                 status):
        
        super().__init__(unique_id, model)
        self.position = position
        self.dimension = len(position)
        self.diffusion_coef = diffusion_coef
        self.status = status

class Binary_image:
    """
     Creating KD_tree from coordinates of ECS in a cubic tissue segment
     Args:
        free_coords: all non-zero pixels in 3D image stacks
        lower_bound: lowest coordinate value (default should be 0)
        upper_bound: largest coordinate value (length of cube in pixels)
    """
    def __init__(self, 
                 free_coords, 
                 lower_bound, 
                 upper_bound):
        
        self.number_free_coords, self.dim = np.shape(free_coords)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.kd_tree = cKDTree(free_coords)

        mask_coord = (free_coords == lower_bound) | (free_coords == upper_bound)
        boundary_indices = np.unique(np.argwhere(mask_coord)[:,0])
        self.boundary_coord_indices = boundary_indices

        del free_coords

    def approximate_equilirium_flux(self, position):
        """If particle went out of bounds, reinsert near the region of exit"""
        # Replace values lower than lower_bound with upper_bound in-place
        position[position < self.lower_bound] = self.lower_bound
        # Replace values higher than upper_bound with lower_bound in-place
        position[position >= self.upper_bound] = self.upper_bound

    def nearest_free_space(self, position):
        """Place particle at a nearest available ECS position"""
        _, index = self.kd_tree.query(position)
        return self.kd_tree.data[index]

class FRAP_Model(mesa.Model):
    """
    Model for FRAP experiment
    Args:
        N (int): total number of particles
        binary_image (Binary_image)
        dt (float): time step size
        D (float): true diffusion coefficient of particles
        frap_center (array): coordinate of center of photobleaching region
        frap_R (float): radius of photobleaching region
        save_particles (binary): True to save particle positions
    """
    def __init__(self,
                 N,
                 binary_image,
                 dt,
                 D,
                 frap_center,
                 frap_R,
                 save_particles):
        
        self.N = N
        self.binary_image = binary_image
        self.dt = dt
        self.D = D
        self.frap_center = frap_center
        self.frap_R = frap_R
        self.schedule = mesa.time.RandomActivation(self)

        # create initial conditions
        # uniformly distribute N particles onto available positions in image
        total_space = self.binary_image.number_free_coords
        rand_pos = np.random.randint(low=0, high=total_space, size=self.N)
        for i in range(self.N):
            rand_position = self.binary_image.kd_tree.data[rand_pos[i]]
            in_frap = np.linalg.norm(rand_position - self.frap_center) < self.frap_R
            if in_frap:
                # if inside the frap region turn status "off"
                p = Particle(unique_id=i, 
                             model=self,
                             position=rand_position,
                             diffusion_coef=self.D,
                             status="off")
                self.schedule.add(p)
            else:
                p = Particle(unique_id=i, 
                             model=self,
                             position=rand_position,
                             diffusion_coef=self.D,
                             status="on")
                self.schedule.add(p)
        
        # to save data from agent-based model
        if save_particles:
            self.datacollector = mesa.DataCollector(model_reporters={"frap_region":compute_off_on},
                                                    agent_reporters={"Position":"position"})
        else:
            self.datacollector = mesa.DataCollector(model_reporters={"frap_region":compute_off_on})

    def step(self, save_data):
        # only collect data in this iteration if save_data==True
        if save_data:
            self.datacollector.collect(self)

        # collect positions of all particles
        all_positions = np.array([agent.position for agent in self.schedule.agents])

        # compute diffusion in batches
        D = self.D
        dt = self.dt
        gaussian_samples = np.sqrt(2*D*dt) * np.random.randn(self.N, self.binary_image.dim)
        new_positions = all_positions + gaussian_samples.astype(int)
        self.binary_image.approximate_equilirium_flux(new_positions)
        new_positions = self.binary_image.nearest_free_space(new_positions)

        # update position of agents
        for agent, new_position in zip(self.schedule.agents, new_positions):
            agent.position = new_position

        self.schedule.step()


class FCS_Model(mesa.Model):
    """
    Model for FCS simulations
    Args:
        N (int): total number of particles
        binary_image (Binary_image)
        dt (float): time step size
        D (float): true diffusion coefficient of particles
        fcs_center (array): coordinate of center of confocal region
        fcs_R (float): radius of confocal region, must be small!
        save_particles (binary): True to save particle positions
    """
    def __init__(self,
                 N,
                 binary_image,
                 dt,
                 D,
                 fcs_center,
                 fcs_R, 
                 save_particles):
        
        self.N = N
        self.binary_image = binary_image
        self.dt = dt
        self.D = D
        self.fcs_center = fcs_center
        self.fcs_R = fcs_R
        self.schedule = mesa.time.RandomActivation(self)

        # create initial conditions
        # uniformly distribute N particles onto available positions in image
        total_space = self.binary_image.number_free_coords
        rand_pos = np.random.randint(low=0, high=total_space, size=self.N)
        for i in range(self.N):
            rand_position = self.binary_image.kd_tree.data[rand_pos[i]] # array type
            p = Particle(unique_id=i, 
                         model=self,
                         position=rand_position,
                         diffusion_coef=self.D,
                         status="on")
            self.schedule.add(p)

        # to save data from agent-based model
        if save_particles:
            self.datacollector = mesa.DataCollector(model_reporters={"fcs_region":count_total_in_fcs},
                                                    agent_reporters={"Position":"position"})
        else:
            self.datacollector = mesa.DataCollector(model_reporters={"fcs_region":count_total_in_fcs})

    def step(self, save_data=True):
        # only collect data in this iteration if save_data==True
        if save_data:
            self.datacollector.collect(self)

        # collect positions of all particles
        all_positions = np.array([agent.position for agent in self.schedule.agents])

        # compute diffusion in batches
        D = self.D
        dt = self.dt
        gaussian_samples = np.sqrt(2*D*dt) * np.random.randn(self.N, self.binary_image.dim)
        new_positions = all_positions + gaussian_samples.astype(int)
        self.binary_image.approximate_equilirium_flux(new_positions)
        new_positions = self.binary_image.nearest_free_space(new_positions)

        # update agents
        for agent, new_position in zip(self.schedule.agents, new_positions):
            agent.position = new_position

        self.schedule.step()