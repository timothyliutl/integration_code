import numpy as np
from functions import inner_int_matrix, convolution, convolution2, convolution_w1_w2, inner_int_division_matrix, inner_int_matrix_div, gpu_sum_reduce
import math
import pandas as pd
from numba import cuda

# x_val = np.arange(-12,12,0.02)
# y_val = np.arange(-8,8,0.02)
# z_val = np.arange(-8,8,0.02)


# threadsperblock = (10, 10, 10)
# blockspergrid_x = math.ceil(x_val.shape[0] / threadsperblock[0])
# blockspergrid_y = math.ceil(x_val.shape[0] / threadsperblock[1])
# blockspergrid_z = math.ceil(x_val.shape[0] / threadsperblock[2])
# blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)


df = pd.DataFrame(columns=['alpha', 'value'])

# threadsperblock_2d = (32,32)
# blockspergrid_x = math.ceil(x_val.shape[0] / threadsperblock[0])
# blockspergrid_y = math.ceil(x_val.shape[0] / threadsperblock[1])
# blockspergrid_2d = (blockspergrid_x, blockspergrid_y)

convolution_x_val = np.arange(-40,40,0.02)
t_val = np.arange(-40,40,0.02)
convolution_array_yw1 = convolution(convolution_x_val, t_val, 0.02)

convolution_x_val = np.arange(-20,20,0.02)
t_val = np.arange(-20,20,0.02)
convolution_array_yw1w2 = convolution2(convolution_x_val, t_val, convolution_array_yw1, 0.02)
convolution_array_yw1 = convolution(convolution_x_val, t_val, 0.02)
convolution_array_w1w2 = convolution_w1_w2(convolution_x_val, t_val, 0.02)

x_val = np.arange(-12,12,0.02)
y_val = np.arange(-6,6,0.02)
z_val = np.arange(-6,6,0.02)

threadsperblock = (10, 10, 10)
blockspergrid_x = math.ceil(x_val.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(x_val.shape[0] / threadsperblock[1])
blockspergrid_z = math.ceil(x_val.shape[0] / threadsperblock[2])
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

threadsperblock_2d = (32,32)
blockspergrid_x = math.ceil(x_val.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(x_val.shape[0] / threadsperblock[1])
blockspergrid_2d = (blockspergrid_x, blockspergrid_y)

alpha = 0.5
#for alpha in [0.99]:
result_matrix = np.zeros((x_val.shape[0],y_val.shape[0],z_val.shape[0]))
#result_matrix = cuda.to_device(result_matrix)
inner_int_matrix_div[blockspergrid, threadsperblock](x_val, y_val, z_val, 0.02, convolution_x_val[0] ,alpha, convolution_array_w1w2, convolution_array_yw1w2, convolution_array_yw1, result_matrix)
#np.savetxt("fx_yz.csv", result_matrix[:,500,500], delimiter=",")
reduce_matrix = result_matrix.reshape(-1)


final_result = gpu_sum_reduce(reduce_matrix)*(0.02**3)
print(final_result)