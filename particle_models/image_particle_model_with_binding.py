# Model for image-based particle model for FRAP experiments with receptor bindings
import numpy as np
import mesa
import image_based_particle_model as pm

class Particle(mesa.Agent):
    """
    An agent representing a diffusive particle that may be bounded
    Args:
        unique_id (int): unique ID for particle
        model (mesa.Model)
        position (array): array of particle coordinate
        status (string): fluorescent status of particle
        ("on" for fluorescent particles, "off" for bleached particles)
        binding_status (binary): binding status of particles
        (True for bounded particles, False for free particles)
    """
    def __init__(self,
                 unique_id,
                 model,
                 position,
                 diffusion_coef,
                 status,
                 binding_status):
        
        super().__init__(unique_id, model)
        self.position = position
        self.dimension = len(position)
        self.diffusion_coef = diffusion_coef
        self.status = status
        self.binding_status = binding_status

class FRAP_Model(mesa.Model):
    """
    Model for FRAP experiment with receptor bindings
    Args:
        N (int): total number of particles
        binary_image (Binary_image)
        dt (float): time step size
        D (float): true diffusion coefficient of particles
        frap_center (array): coordinate of center of photobleaching region
        frap_R (float): radius of photobleaching region
        R_bound_initial (int): initial number of bounded particles
        k_bound (float): rate of particles being bounded
        k_free (float): rate of particles being released
        save_particles (binary): True to save particle positions
    """

    def __init__(self,
                 N,
                 binary_image,
                 dt,
                 D,
                 frap_center,
                 frap_R,
                 R_bound_initial,
                 k_bound,
                 k_free,
                 save_particles):
        
        self.N = N                              # number of particles
        self.binary_image = binary_image        # tortuous image
        self.dt = dt                            # time step size
        self.D = D                              # diffusion coefficient
        self.frap_center = frap_center          # centre of frap
        self.frap_R = frap_R                    # radius of frap sphere
        self.R_bound_initial = R_bound_initial  # initial number of bounded receptors (<R_total and <N)
        self.k_bound = k_bound                  # binding affinity
        self.k_free = k_free                    # diassociation constant
        self.schedule = mesa.time.RandomActivation(self)

        # create initial conditions
        # uniformly distribute N particles onto available positions in image
        total_space = self.binary_image.number_free_coords
        rand_pos = np.random.randint(low=0, high=total_space, size=self.N)

        # randomly choose R_bound_initial number of particles to be bounded
        rand_bound = np.random.choice(N, R_bound_initial, False)
        binding_status_list = np.array([True if i in rand_bound else False for i in range(N)])

        for i in range(self.N):
            rand_position = self.binary_image.kd_tree.data[rand_pos[i]] # array type
            in_frap = np.linalg.norm(rand_position - self.frap_center) < self.frap_R
            if in_frap:
                # if inside the frap region turn it off
                p = Particle(unique_id=i, 
                             model=self,
                             position=rand_position,
                             diffusion_coef=self.D,
                             status="off",
                             binding_status=binding_status_list[i])
                self.schedule.add(p)
            else:
                p = Particle(unique_id=i, 
                             model=self,
                             position=rand_position,
                             diffusion_coef=self.D,
                             status="on",
                             binding_status=binding_status_list[i])
                self.schedule.add(p)
        
        if save_particles:
            self.datacollector = mesa.DataCollector(model_reporters={"frap_region":pm.compute_off_on},
                                                    agent_reporters={"Position":"position", "Binding":"binding_status"})
        else:
            self.datacollector = mesa.DataCollector(model_reporters={"frap_region":pm.compute_off_on})
    
    def diffuse_step(self, all_positions, all_binding_status):
        """Diffuse and update positions for particles that are unbounded (binding_status==False)"""
        gaussian_samples = np.sqrt(2*self.D*self.dt) * np.random.randn(self.N, self.binary_image.dim)
        # convert binding status to integer
        binding_int = 1 - np.array(all_binding_status).astype(int)
        cover_binding = np.column_stack((binding_int,)*self.binary_image.dim)
        new_positions = all_positions + gaussian_samples.astype(int) * cover_binding
        self.binary_image.approximate_equilirium_flux(new_positions) # if particle went out of bounds, approximate flux in
        new_positions = self.binary_image.nearest_free_space(new_positions) # find nearest space
        return new_positions
    
    def free_bounded_particles(self, bounded_particles):
        """Returns indices of bounded particles that are now freed from receptors"""
        freed_particles = bounded_particles[np.random.rand(len(bounded_particles)) < self.k_free*self.dt]
        return freed_particles

    def bind_free_particles(self, free_particles):
        """Returns indices of free particles that are now bounded to receptors"""
        binding_particles = free_particles[np.random.rand(len(free_particles)) < self.k_bound*self.dt]
        return binding_particles
    
    def make_new_binding_status(self,
                                binding_status, 
                                freed_particles,
                                binding_particles):
        """Updates binding status of particles given indices of particles that are freed and bounded"""
        binding_status[freed_particles] = False
        binding_status[binding_particles] = True
        return binding_status

    def step(self, save_data):
        if save_data:
            self.datacollector.collect(self)

        # collect positions of all particles
        all_positions = np.array([agent.position for agent in self.schedule.agents])
        all_binding_status = np.array([agent.binding_status for agent in self.schedule.agents])

        bound_particle_indices = np.where(all_binding_status)[0] # indices of bounded particles
        free_particle_indices = np.where(~all_binding_status)[0] # indices of free particles

        # Compute diffusion in batches
        new_positions = self.diffuse_step(all_positions, all_binding_status)
        # Randomly free bounded particles
        freed_particles = self.free_bounded_particles(bound_particle_indices)
        # Randomly bind free particles
        binding_particles = self.bind_free_particles(free_particle_indices)
        # Update binding status
        new_binding_status = self.make_new_binding_status(all_binding_status, freed_particles, binding_particles)

        # update agents
        for agent, new_position, binding in zip(self.schedule.agents, new_positions, new_binding_status):
            agent.position, agent.binding_status = new_position, binding

        self.schedule.step()