import numpy as np
from functions import inner_int_matrix, convolution, convolution2
import math

x_val = np.arange(-5,5,0.01)
y_val = np.arange(-5,5,0.01)
z_val = np.arange(-5,5,0.01)

result_matrix = np.zeros((x_val.shape[0],x_val.shape[0],x_val.shape[0]))

threadsperblock = (10, 10, 10)
blockspergrid_x = math.ceil(x_val.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(x_val.shape[0] / threadsperblock[1])
blockspergrid_z = math.ceil(x_val.shape[0] / threadsperblock[2])
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)


convolution_x_val = np.arange(-10,10,0.01)
t_val = np.arange(-5,5,0.01)
convolution_array = convolution(convolution_x_val, t_val, 0.01)

inner_int_matrix[blockspergrid, threadsperblock](x_val, y_val, z_val, convolution_array, result_matrix)

convolution_array2 = convolution2(x_val, t_val, convolution_array, 0.01)
#np.savetxt("foo.csv", convolution_array2, delimiter=",")
print(convolution_array2[0:10])