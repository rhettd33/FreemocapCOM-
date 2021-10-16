from __future__ import division
import numpy as np 
import time
import math
import numpy as np
from numpy.lib.npyio import savetxt
import scipy.io
from numpy.core.numeric import NaN


data_array = np.load('octoberfifteen.npy')

end = time.time()

#print("\nData summary:\n", data_array)
print("\nData shape:\n", data_array.shape)

f = []



data_array2 = data_array[:,0:45,:]

data_parse1 = data_array[:,10,:]
data_parse2 = data_array[:,11,:]
data_parse3 = data_array[:,12,:]
data_parse4 = data_array[:,14,:]
data_parse5 = data_array[:,15,:]
data_parse6 = data_array[:,16,:]
data_parse7 = data_array[:,17,:]
data_parse8 = data_array[:,32,:]
data_parse9 = data_array[:,25,:]
data_parse10 = data_array[:,26,:]


savetxt('data2.csv', data_parse1)
savetxt('data3.csv', data_parse2)
savetxt('data4.csv', data_parse3)
savetxt('data5.csv', data_parse4)
savetxt('data6.csv', data_parse5)
savetxt('data7.csv', data_parse6)
savetxt('data8.csv', data_parse7)
savetxt('data9.csv', data_parse8)
savetxt('data10.csv', data_parse9)
savetxt('data11.csv', data_parse10)


# def average_points(a):
#     avgs = []
#     i = 0 
#     while i < 3:
#         t = 0 
#         bad = 0 
#         for point in a:
#             if np.isnan(point[i]):
#                 bad = bad + 1
#             else:
#                 t = t + point[i]
       
#         if len(a) == bad:
#             t = 0
#         else:
            
#             t = t/(len(a) - bad)
            
#         avgs.append(t)
#         i = i + 1

#     return avgs

# total_array = map(lambda x: average_points(x), data_array2)

# print(total_array)
# result2 = np.array(total_array)
# print("Data shape:", result2.shape)

# new_col = np.sum(result2,1).reshape((result2.shape[0],1))
# np.append(result2,new_col,1)

# result3 = np.array([0  ,0, 0])
# print(result3)
# savetxt('data.csv', result2)
# i = 1

# for row in total_array:
#     row2 = np.append(row, [i])
#     print(row2)
#     i = i + 1
#     #result3 = np.vstack([result3, row2])

# print(result3)

