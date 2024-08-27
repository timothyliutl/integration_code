import numpy as np
from functions import inner_int_matrix, convolution, convolution2, convolution_w1_w2, inner_int_division_matrix
import math
import pandas as pd

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

convolution_x_val = np.arange(-40,40,0.005)
t_val = np.arange(-40,40,0.005)
convolution_array_yw1 = convolution(convolution_x_val, t_val, 0.005)

convolution_x_val = np.arange(-20,20,0.005)
t_val = np.arange(-20,20,0.005)
convolution_array_yw1w2 = convolution2(convolution_x_val, t_val, convolution_array_yw1, 0.005)
convolution_array_yw1 = convolution(convolution_x_val, t_val, 0.005)
convolution_array_w1w2 = convolution_w1_w2(convolution_x_val, t_val, 0.005)



for alpha in np.arange(0.8,1,0.01):
    sum_val = 0
    for i in range(8):
        for j in range(8):

            x_val = np.arange(-12,12,0.005)
            y_val = np.arange(2*i-8,2*i-6,0.005)
            z_val = np.arange(2*j-8,2*j-6,0.005)

            threadsperblock = (10, 10, 10)
            blockspergrid_x = math.ceil(x_val.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(x_val.shape[0] / threadsperblock[1])
            blockspergrid_z = math.ceil(x_val.shape[0] / threadsperblock[2])
            blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

            threadsperblock_2d = (32,32)
            blockspergrid_x = math.ceil(x_val.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(x_val.shape[0] / threadsperblock[1])
            blockspergrid_2d = (blockspergrid_x, blockspergrid_y)


        #for alpha in [0.99]:
            result_matrix = np.zeros((x_val.shape[0],y_val.shape[0],z_val.shape[0]))
            print('calculating for alpha, j, i = ', alpha, j, i)
            inner_int_matrix[blockspergrid, threadsperblock](x_val, y_val, z_val, 0.005, convolution_x_val[0] ,alpha, convolution_array_w1w2, convolution_array_yw1w2, convolution_array_yw1, result_matrix)
            #np.savetxt("fx_yz.csv", result_matrix[:,500,500], delimiter=",")
            result_matrix = np.sum((result_matrix*(0.005)), axis=1)
            print(result_matrix[300:350, 300:350])

            
            # new_row = pd.DataFrame({'alpha': [alpha], 'value': [(result_matrix[0,0])]})
            # df = pd.concat([df, new_row])
            # df.to_csv('data-int4.csv')

            final_result = np.zeros(result_matrix.shape)
            inner_int_division_matrix[blockspergrid_2d, threadsperblock_2d](y_val, z_val, 0.005, convolution_x_val[0], result_matrix, convolution_array_w1w2, final_result)
            #np.savetxt("pdf.csv", np.sum(final_result* ((0.02)), axis=1), delimiter=",")
            #final_result = final_result * ((0.02)**2)
            #value somehow negative



            print(np.sum(final_result)/(alpha-1))
            sum_val = sum_val + np.sum(final_result)
    new_row = pd.DataFrame({'alpha': [alpha], 'value': [sum_val/(alpha-1)]})
    df = pd.concat([df, new_row])
    df.to_csv('data_modified_5.csv')
