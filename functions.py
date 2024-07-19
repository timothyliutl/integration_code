from numba import jit
from numba import cuda
import numpy as np

@cuda.jit(device=True)
def exp_dist(x):
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)

@cuda.jit(device=True)
def dist_y(y):
    return np.exp(-y**2/2)/np.sqrt(2*np.pi)

@cuda.jit(device=True)
def f_x_given_y_z(x,y,z):
    return (exp_dist(z-x)*exp_dist(x-y)/(1)) #NOTE: include convolution later

@cuda.jit
def inner_int_matrix(x_array, y_array, z_array, return_matrix):
    x_pos, y_pos, z_pos = cuda.grid(3)
    if x_pos < len(x_array) and y_pos < len(y_array) and z_pos < len(z_array):
        return_matrix[x_pos, y_pos, z_pos] = f_x_given_y_z(x_pos, y_pos, z_pos)
