from numba import jit
from numba import cuda
import numpy as np
import math


@jit(nopython=True)
def convolution(x_val, t_val, step_size):
    #convlution of y w1
    return_array = np.zeros(len(x_val))

    def dist_y(y):
        return math.exp(-y**2/2)/math.sqrt(2*np.pi)
    def dist_w1(x):
        return math.exp(-x**2/2)/math.sqrt(2*np.pi)
    for x_index in range(len(x_val)):
        sum_val = 0
        for t in t_val:
            sum_val += dist_y(x_val[x_index]-t)*dist_w1(t)*step_size
        return_array[x_index] = sum_val
    return return_array

@jit(nopython=True)
def convolution_w1_w2(x_val, t_val, step_size):
    return_array = np.zeros(len(x_val))

    def dist_w2(y):
        return math.exp(-y**2/2)/math.sqrt(2*np.pi)
    def dist_w1(x):
        return math.exp(-x**2/2)/math.sqrt(2*np.pi)
    for x_index in range(len(x_val)):
        sum_val = 0
        for t in t_val:
            sum_val += dist_w2(x_val[x_index]-t)*dist_w1(t)*step_size
        return_array[x_index] = sum_val
    return return_array

@jit(nopython=True)
def convolution2(x_val, t_val, convolution_array, step_size):
    #convolution yw1w2
    return_array = np.zeros(len(x_val))

    def dist_w2(x):
        return math.exp(-x**2/2)/math.sqrt(2*np.pi)
    for x_index in range(len(x_val)):
        sum_val = 0
        for t in t_val:
            sum_val += convolution_array[int((x_val[x_index]-t+10)/0.01)]*dist_w2(t)*step_size
        return_array[x_index] = sum_val
    return return_array


@cuda.jit(device=True)
def exp_dist(x):
    return math.exp(-x**2/2)/math.sqrt(2*np.pi)

@cuda.jit(device=True)
def dist_y(y):
    return math.exp(-y**2/2)/math.sqrt(2*np.pi)

@cuda.jit(device=True)
def f_x_given_y_z(x,y,z, conv_val):
    return (exp_dist(z-x)*exp_dist(x-y)/(conv_val)) #NOTE: include convolution later

@cuda.jit(device=True)
def f_x_given_z(x,z, conv_val_y1w1, conv_val_yw1w2):
    return exp_dist(z-x)*conv_val_y1w1/conv_val_yw1w2

@cuda.jit
def inner_int_matrix(x_array, y_array, z_array, alpha, convolution_array_w1w2, convolution_array_w1w2y, convolution_array_w1y, return_matrix):
    x_pos, y_pos, z_pos = cuda.grid(3)
    if x_pos < len(x_array) and y_pos < len(y_array) and z_pos < len(z_array):
        funct1_val = f_x_given_y_z(x_array[x_pos], y_array[y_pos], z_array[z_pos], convolution_array_w1w2[int((y_array[y_pos] - z_array[z_pos]+10)/0.01)])**alpha
        funct2_val = f_x_given_z(x_array[x_pos], z_array[z_pos], convolution_array_w1y[int((x_array[x_pos]+10)/0.01)], convolution_array_w1w2y[int((z_array[z_pos]+10)/0.01)])**(1-alpha)
        return_matrix[x_pos, y_pos, z_pos] =  funct1_val*funct2_val

@cuda.jit
def inner_int_division_matrix(y_array, z_array, inner_int_matrix, convolution_array_w1w2, return_matrix):
    y_pos, z_pos = cuda.grid(2)
    if y_pos< len(y_array) and z_pos < len(z_array):
        f_y_z_val = convolution_array_w1w2[int((z_array[z_pos]- y_array[y_pos]+10)/0.01)] * dist_y(y_array[y_pos])
        return_matrix[y_pos, z_pos] = inner_int_matrix[y_pos, z_pos] * f_y_z_val