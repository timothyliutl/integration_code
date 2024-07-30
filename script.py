import numpy as np
from functions import inner_int_matrix, convolution, convolution2, convolution_w1_w2, inner_int_division_matrix
import math
import pandas as pd

x_val = np.arange(-5,5,0.01)
y_val = np.arange(-5,5,0.01)
z_val = np.arange(-5,5,0.01)


threadsperblock = (10, 10, 10)
blockspergrid_x = math.ceil(x_val.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(x_val.shape[0] / threadsperblock[1])
blockspergrid_z = math.ceil(x_val.shape[0] / threadsperblock[2])
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)


convolution_x_val = np.arange(-10,10,0.01)
t_val = np.arange(-5,5,0.01)
convolution_array_yw1 = convolution(convolution_x_val, t_val, 0.01)
convolution_array_yw1w2 = convolution2(convolution_x_val, t_val, convolution_array_yw1, 0.01)
#np.savetxt("foo.csv", convolution_array2, delimiter=",")
convolution_array_w1w2 = convolution_w1_w2(convolution_x_val, t_val, 0.01)

df = pd.DataFrame(columns=['alpha', 'value'])

for alpha in np.arange(0.001,0.999, 0.01):
    result_matrix = np.zeros((x_val.shape[0],x_val.shape[0],x_val.shape[0]))
    print('calculating for alpha = ', alpha)
    inner_int_matrix[blockspergrid, threadsperblock](x_val, y_val, z_val,alpha, convolution_array_w1w2, convolution_array_yw1w2, convolution_array_yw1, result_matrix)
    result_matrix = np.sum(result_matrix, axis=0)* 0.01
    def myfunc(x):
        return math.log10(x)
    vfunc = np.vectorize(myfunc)
    result_matrix = vfunc(result_matrix)
    print(result_matrix[0:5,0:5])

    final_result = np.zeros(result_matrix.shape)
    inner_int_division_matrix[blockspergrid, threadsperblock](y_val, z_val, result_matrix, convolution_array_w1w2, final_result)
    final_result = final_result * ((0.01)**2)
    print(final_result[0:5,0:5])
    print(np.sum(final_result)/(alpha-1))
    new_row = pd.DataFrame({'alpha': [alpha], 'value': [np.sum(final_result)/(alpha-1)]})
    df = pd.concat([df, new_row])

    df.to_csv('data.csv')
