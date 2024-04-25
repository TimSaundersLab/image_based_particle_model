import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import gc
from scipy.optimize import curve_fit
from scipy.special import i0
from scipy.special import i1

# PREPARING 3D IMAGES FOR SIMULATIONS
def get_dimensions(directory):
    """ Getting dimensions of image stacks given folder directory """
    stacks = sorted(os.listdir(directory))
    first_image_path = os.path.join(directory, stacks[0])
    with Image.open(first_image_path) as first_image:
        row, col = np.array(first_image).shape
    return row, col

def make_3D_image(directory, filename, dtype=np.uint8):
    """
    Make a 3D image array from a folder containing Z-stacks
    Args:
        directory (string): file path containing image stack
        filename (string): temporary filename (E.g., use os.path.join(mkdtemp(), 'output_3d_array.npy'))
    """
    row, col = get_dimensions(directory)
    image_list = sorted(os.listdir(directory))
    shape = (len(image_list), row, col)
    memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)

    for i in tqdm(range(len(image_list))):
        tiff_file_path = os.path.join(directory, image_list[i])
        image_data = np.array(Image.open(tiff_file_path))
        memmap_array[i, :, :] = image_data
        del image_data
    
    memmap_array.flush()
    return memmap_array

def get_3D_indices(image, filename, dtype=np.int32):
    """
    Obtain indices of all extracellular space pixels in 3D image
    Args:
        image (array): array of 3D image
        filename: (string): temporary filename (E.g., use os.path.join(mkdtemp(), '3d_indices.npy'))
    """
    num_image,_,_ = np.shape(image)
    non_zeros = np.count_nonzero(image)
    ind_shape = (non_zeros, 3)
    ind_memmap = np.memmap(filename, dtype=dtype, mode='w+', shape=ind_shape)

    counter = 0
    for i in tqdm(range(num_image)):
        chunk = image[i, :, :]
        chunk_indices = np.array(np.where(chunk != 0)).T
        count, _ = np.shape(chunk_indices)
        z_coord = np.ones(count)*i
        indices = np.hstack((z_coord[:, np.newaxis], chunk_indices))
        ind_memmap[counter:counter+count, :] = indices
        counter += count
        del chunk
    
    ind_memmap.flush()
    return ind_memmap

def position_to_folder(data_collector_df, 
                       folder_path, 
                       dimension, 
                       status_list):
    """
    Save particle positions into individual folder for each time step with csv files (for Paraview visualisation)
    Args:
        data_collector_df (pandas): dataframe from mesa data collector
        folder_path (string): folder to save csv files in
        dimension (int): dimension of particle simulation
        status_list (list of strings): list of particle status "off" or "on" (for FRAP simulation)
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    steps_array = np.unique(data_collector_df.index.get_level_values("Step").to_numpy()) # iteration numbers
    for i in range(len(steps_array)):
        all_pos = np.array(data_collector_df.xs(steps_array[i], level="Step")["Position"])
        pos_array = np.stack(all_pos, axis=0)
        x_pos = pos_array[:,0]
        y_pos = pos_array[:,1]

        if dimension==2:
            # make dataframe
            data = {"x":x_pos, "y":y_pos, "status":status_list}
            step_df = pd.DataFrame(data)
            step_df.to_csv(f"{folder_path}/iter_{steps_array[i]}.csv")
            del x_pos, y_pos, data, step_df

        else:
            z_pos = pos_array[:,2]
            del all_pos, pos_array
            
            # make dataframe
            data = {"x":x_pos, "y":y_pos, "z":z_pos, "status":status_list}
            step_df = pd.DataFrame(data)
            step_df.to_csv(f"{folder_path}/iter_{steps_array[i]}.csv")
            del x_pos, y_pos, z_pos, data, step_df

        gc.collect()

def position_to_folder_with_binding(data_collector_df, 
                                    folder_path, 
                                    dimension, 
                                    status_list):
    """
    Save particle positions into individual folder for each time step with csv files 
    (for simulations with binding)
    (for Paraview visualisation)
    Args:
        data_collector_df (pandas): dataframe from mesa data collector
        folder_path (string): folder to save csv files in
        dimension (int): dimension of particle simulation
        status_list (list of strings): list of particle status "off" or "on" (for FRAP simulation)
    """
    # make folder if folder doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    steps_array = np.unique(data_collector_df.index.get_level_values("Step").to_numpy()) # iteration numbers
    for i in range(len(steps_array)):
        all_pos = np.array(data_collector_df.xs(i, level="Step")["Position"])
        binding_list = np.array(data_collector_df.xs(i, level="Step")["Binding"])
        pos_array = np.stack(all_pos, axis=0)
        x_pos = pos_array[:,0]
        y_pos = pos_array[:,1]

        if dimension==2:
            # make dataframe
            data = {"x":x_pos, "y":y_pos, "status":status_list, "binding":binding_list}
            step_df = pd.DataFrame(data)
            step_df.to_csv(f"{folder_path}/iter_{i}.csv") # save as csv
            del x_pos, y_pos, data, step_df
        else:
            z_pos = pos_array[:,2]
            # make dataframe
            data = {"x":x_pos, "y":y_pos, "z":z_pos, "status":status_list, "binding":binding_list}
            step_df = pd.DataFrame(data)
            step_df.to_csv(f"{folder_path}/iter_{i}.csv") # save as csv
            del x_pos, y_pos, z_pos, data, step_df
        
        gc.collect()

# EXTRACTING DATA FROM FRAP SIMULATIONS
def convert_string_to_array(string_example):
    """Take string e.g. "[2 3]" to 1D numpy array"""
    numbers = string_example.strip('[]').split()
    int_numbers = [float(num) for num in numbers]
    numpy_array = np.array(int_numbers)
    return numpy_array

def get_frap_data(path, nt):
    """
    Obtain FRAP results (on/total particles in FRAP region) from a single file
    Args:
        path (string): path to the file
        nt (int): number of simulation time steps to obtain, max is the length of the dataframe in the path
    """
    df = pd.read_csv(f'{path}')
    off_on = np.array(df["frap_region"])[:nt]
    off_on = [convert_string_to_array(i) for i in off_on]
    off_num = np.array([frap[0] for frap in off_on])
    on_num = np.array([frap[1] for frap in off_on])
    return off_num, on_num #frap_data

def get_frap_mean_and_std(path):
    """
    Obtain FRAP results (on/total particles in FRAP region) from all files in path
    Args:
        path (string): path containing files 
        time_steps (int): number of time steps in FRAP simulations (<=length of dataframes in files)
    """
    data_files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))])
    nt = len(pd.read_csv(os.path.join(path, data_files[0]))) # number of simulation time steps in the files
    ns = len(data_files) # number of sample simulations in path
    data = np.zeros((ns, nt))
    for i in range(ns):
        file_path = f'{path}/{data_files[i]}'
        off_num, on_num = get_frap_data(file_path, nt)
        data[i, :] = on_num / (off_num + on_num)
    mean = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    return mean, sd

def get_total_frap(path, particles_per_file=20000):
    """
    Obtain FRAP results (on/total particles in FRAP region) from all files in path, by getting the total particles
    Args:
        path (string): path containing files 
        particles_per_file (int): number of particles per simulation
    """
    data_files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))])
    nt = len(pd.read_csv(os.path.join(path, data_files[0]))) # number of entries in dataset
    ns = len(data_files) # number of sample simulations in path
    total_particles = particles_per_file * ns
    all_on = np.zeros((ns, nt))
    all_off = np.zeros((ns, nt))
    for i in range(ns): # loop through all sample files to get data
        file_path = f'{path}/{data_files[i]}'
        all_off[i, :], all_on[i, :] = get_frap_data(file_path, nt)
    total_on = np.sum(all_on, axis=0)
    total_off = np.sum(all_off, axis=0)
    data = total_on / (total_on + total_off) #total_off[0]
    finf = 1 - (total_off[0] / total_particles)
    return data, finf # total_on, total_off

def get_total_particles_in_frap(path):
    """
    Obtain total number of particles in FRAP region over time
    Args:
        path (string): path containing files
    """
    data_files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))])
    nt = len(pd.read_csv(os.path.join(path, data_files[0]))) # number of entries in dataset
    ns = len(data_files) # number of sample simulations in path
    all_on = np.zeros((ns, nt))
    all_off = np.zeros((ns, nt))
    for i in range(ns): # loop through all sample files to get data
        file_path = f'{path}/{data_files[i]}'
        all_off[i, :], all_on[i, :] = get_frap_data(file_path, nt)
    total_on = np.sum(all_on, axis=0)
    total_off = np.sum(all_off, axis=0)
    data = total_on + total_off
    return data 

def get_no_cells_frap_data(path):
    """Obtain FRAP results from a single csv file (for no cells simulations)"""
    df = pd.read_csv(path)
    all_off, all_on = df["off_num"].to_numpy(), df["on_num"].to_numpy()
    data = all_on / (all_on + all_off)
    return data

def get_no_cells_total_frap(path, particles_per_file=20000):
    """
    Obtain FRAP results (on/total particles in FRAP region) from all files in path, by getting the total particles (for no cells simulation)
    Args:
        path (string): path containing files 
        particles_per_file (int): number of particles per simulation
    """
    all_files = os.listdir(path)
    nt = len(pd.read_csv(os.path.join(path, all_files[0])))
    ns = len(all_files)
    total_particles = particles_per_file * ns
    all_on = np.zeros((ns, nt))
    all_off = np.zeros((ns, nt))
    for i in range(ns):
        file_path = f'{path}/{all_files[i]}'
        df = pd.read_csv(file_path)
        all_off[i, :], all_on[i, :] = df["off_num"], df["on_num"]
    total_on = np.sum(all_on, axis=0)
    total_off = np.sum(all_off, axis=0)
    data = total_on / (total_off+total_on)
    finf = 1 - (total_off[0] / total_particles)
    return data, finf 

### DATA ANALYSIS FOR FRAP RESULTS
def soumpasis_function(time, tau, f_inf):
    """ 
    Soumpasis function for diffusion only or diffusion dominated FRAP
    Args:
        time (float/array): time after photobleaching
        tau (float): characteristic diffusion time
        f_inf (float): stable value after recovery
    """
    return f_inf * np.exp(-(2*tau)/time) * (i0((2*tau)/time) + i1((2*tau)/time))

def fit_soumpasis_curve(data, time, f_inf=1):
    """
    Fit Soumpasis Curve to FRAP data
    Args:
        data (array): FRAP data after photobleaching
        time (array): FRAP experiment time
        f_inf (float): stable state after FRAP
    """
    def soumpasis_fit(time, tau):
        return f_inf * np.exp(-(2*tau)/time) * (i0((2*tau)/time) + i1((2*tau)/time))
    params, cov = curve_fit(soumpasis_fit, time, data)
    return params[0]

def exponential_fit(time, C, tau):
    """ 
    Exponential function
    Args:
        time (float/array): time after photobleaching
        C (float): constant
        tau (float):  characteristic diffusion time
    """
    return 1 - C * np.exp(-tau *time)

def fit_exponential_curve(data, time):
    """
    Fit Exponential Curve to FRAP data
    Args:
        data (array): FRAP data after photobleaching
        time (array): FRAP experiment time
    """
    params, cov = curve_fit(exponential_fit, time, data)
    return params[0], params[1]

def diffusion_coef(tau, k, w):
    """
    To calculate diffusion coefficient from time constant
    Args:
        tau (float): diffusion time constant
        k (float): constant of proportionality, for 2D circular disc, k=4
        w (float): radius of FRAP region
    """
    return (w**2) / (k*tau)

def fit_no_cells_case(tau_fitted, true_diff, w):
    """
    Find the constant from no cells data
    Args:
        tau_fitted (array): diffusion time constants
        true_diff (array): ground truth diffusion coefficients
        w (float): radius of photobleaching region
    """
    def diff_coef(tau, k):
        return (w**2) / (k*tau)
    param, _ = curve_fit(diff_coef, tau_fitted, true_diff)
    return param[0]

def linear(X, k):
    """Linear function"""
    return k*X

def fit_linear(x, data):
    """Fitting a linear function"""
    param, cov = curve_fit(linear, x, data)
    return param[0]

### DATA ANALYSIS FOR FCS RESULTS
def G(lag_time, tau, G0):
    """
    Theoretical autocorrelation function for 3D diffusion model
    Args:
        lag_time (float/array): lag time of autocorrelation
        tau (float): diffusion time constant
        G0 (float): mean number of particles in confocal volume during FCS
    """
    return (G0 * ((1 + (lag_time/tau))**(-1)) * ((1 + (lag_time/tau))**(-1/2)))

def fit_fcs_curve(G0, lag_time, data):
    """
    Fit FCS data for diffusion time constant with theoretical autocorrelation function
    Args:
        G0 (float): mean number of particles in confocal volume during FCS
        lag_time (float/array): lag time of autocorrelation
        data (array): Autocorrelation from FCS time series
    """
    def G(lag_time, tau):
        return (G0 * ((1 + (lag_time/tau))**(-1)) * ((1 + (lag_time/tau))**(-1/2)))
    params, cov = curve_fit(G, lag_time, data)
    return params[0]

def autocorr_fcs(data):
    """Autocorrelation function for FCS time series"""
    centered_data = data - np.mean(data)
    acf_values = np.correlate(centered_data, centered_data, mode="full") / len(data)
    return acf_values[len(data)-1:] / (np.mean(data)**2) # normalise by mean