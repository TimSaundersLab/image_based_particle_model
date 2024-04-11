import numpy as np
from scipy import spatial as sp
from scipy.spatial import cKDTree
import mesa
from tqdm import tqdm

def count_total_in_fcs(model):
    agents_in_fcs = [agent for agent in model.schedule.agents if np.linalg.norm(agent.position - model.fcs_center)<model.fcs_R]
    return len(agents_in_fcs)

def compute_off_on(model):
    # if particle has dist(position, frap_center)<frap_R, count number of on and off
    agents_in_frap = [agent for agent in model.schedule.agents if np.linalg.norm(agent.position - model.frap_center)<model.frap_R]
    off_agents = [agent for agent in agents_in_frap if agent.status=="off"]
    on_agents = [agent for agent in agents_in_frap if agent.status=="on"]
    return np.array([len(off_agents), len(on_agents)])

class Particle(mesa.Agent):
    """ An agent representing a diffusive particle"""
    def __init__(self,
                 unique_id,
                 model,
                 position,
                 diffusion_coef,
                 status):
        
        super().__init__(unique_id, model)
        self.position = position # array
        self.dimension = len(position) # integer
        self.diffusion_coef = diffusion_coef # float
        self.status = status # "on" or "off" for FRAP

class Binary_image:
    
    def __init__(self, 
                 free_coords, 
                 lower_bound, 
                 upper_bound):
        """
        Creating binary image with sparse matrix
        free_coords: all non-zero pixels in 3D image stacks
        sparse: sparse matrix
        obstacle_label: must be "0" (denotes spaces occupied by cells)
        """
        self.number_free_coords, self.dim = np.shape(free_coords)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.kd_tree = cKDTree(free_coords)

        mask_coord = (free_coords == lower_bound) | (free_coords == upper_bound)
        boundary_indices = np.unique(np.argwhere(mask_coord)[:,0])
        self.boundary_coord_indices = boundary_indices

        del free_coords
    
    def approximate_periodic_bc(self, position):
        """If particle went out of bounds, reinsert from the opposite end"""
        # Replace values lower than lower_bound with upper_bound in-place
        position[position < self.lower_bound] = self.upper_bound
        # Replace values higher than upper_bound with lower_bound in-place
        position[position >= self.upper_bound] = self.lower_bound

    def approximate_equilirium_flux(self, position):
        """ if particle went out of bounds, reinsert in the same end"""
        # Replace values lower than lower_bound with upper_bound in-place
        position[position < self.lower_bound] = self.lower_bound
        # Replace values higher than upper_bound with lower_bound in-place
        position[position >= self.upper_bound] = self.upper_bound

    def random_flux(self, position):
        """If particle went out of bounds, reinsert in a random boundary position"""
        mask = (position < self.lower_bound) | (position > self.upper_bound)
        out_of_bounds_indices = np.argwhere(mask)
        rows = np.unique(out_of_bounds_indices[:,0])
        position[rows, :] = self.kd_tree.data[np.random.choice(self.boundary_coord_indices, len(rows)), :]

    def random_reposition(self, position):
        """If particle went out of bounds, reinsert in any random position in the image"""
        mask = (position < self.lower_bound) | (position > self.upper_bound)
        out_of_bounds_indices = np.argwhere(mask)
        rows = np.unique(out_of_bounds_indices[:,0])
        position[rows, :] = self.kd_tree.data[np.random.randint(0, self.number_free_coords, len(rows)), :]

    def nearest_free_space(self, position):
        """Place particle at a nearest available position"""
        # all_dists = sp.distance.cdist(self.free_coords, np.array([position]))
        # nearest = self.free_coords[np.argmin(all_dists)]
        _, index = self.kd_tree.query(position)
        return self.kd_tree.data[index] # nearest position

class FRAP_Model(mesa.Model):
    """Model for FRAP simulations"""

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
        # randomly choose N positions from binary image
        total_space = self.binary_image.number_free_coords
        rand_pos = np.random.randint(low=0, high=total_space, size=self.N)
        for i in range(self.N):
            rand_position = self.binary_image.kd_tree.data[rand_pos[i]] # array type
            in_frap = np.linalg.norm(rand_position - self.frap_center) < self.frap_R
            if in_frap:
                # if inside the frap region turn it off
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
        
        if save_particles:
            # add data collector to get number of "off" particles and number of "on" particles in region
            self.datacollector = mesa.DataCollector(model_reporters={"frap_region":compute_off_on},
                                                    agent_reporters={"Position":"position"})
        else:
            self.datacollector = mesa.DataCollector(model_reporters={"frap_region":compute_off_on})

    def step(self, save_data):
        # only save data in this iteration if save_data==True
        if save_data:
            self.datacollector.collect(self)

        # collect positions of all particles
        all_positions = np.array([agent.position for agent in self.schedule.agents])

        # compute diffusion in batches
        D = self.D
        dt = self.dt
        gaussian_samples = np.sqrt(2*D*dt) * np.random.randn(self.N, self.binary_image.dim)
        new_positions = all_positions + gaussian_samples.astype(int)
        self.binary_image.approximate_equilirium_flux(new_positions) # if particle went out of bounds, approximate flux in
        new_positions = self.binary_image.nearest_free_space(new_positions) # find nearest space

        # update agents
        for agent, new_position in zip(self.schedule.agents, new_positions):
            agent.position = new_position

        self.schedule.step()


class FCS_Model(mesa.Model):
    """Model for FCS simulations"""

    def __init__(self,
                 N,
                 binary_image,
                 dt,
                 D,
                 fcs_center,
                 fcs_R):
        self.N = N
        self.binary_image = binary_image
        self.dt = dt
        self.D = D
        self.fcs_center = fcs_center
        self.fcs_R = fcs_R
        self.schedule = mesa.time.RandomActivation(self)

        # create initial conditions
        # uniformly distribute N particles onto available positions in image
        # randomly choose N positions from binary image
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
                
        # add data collector to get number of "off" particles and number of "on" particles in region
        self.datacollector = mesa.DataCollector(model_reporters={"fcs_region":count_total_in_fcs})

    def step(self):
        self.datacollector.collect(self)

        # collect positions of all particles
        all_positions = np.array([agent.position for agent in self.schedule.agents])

        # compute diffusion in batches
        D = self.D
        dt = self.dt
        gaussian_samples = np.sqrt(2*D*dt) * np.random.randn(self.N, self.binary_image.dim)
        new_positions = all_positions + gaussian_samples.astype(int)
        self.binary_image.approximate_equilirium_flux(new_positions) # if particle went out of bounds, approximate flux in
        new_positions = self.binary_image.nearest_free_space(new_positions) # find nearest space

        # update agents
        for agent, new_position in zip(self.schedule.agents, new_positions):
            agent.position = new_position

        self.schedule.step()