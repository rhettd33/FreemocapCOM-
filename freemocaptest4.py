from __future__ import division
import matplotlib
import numpy as np 
import time
import math
import numpy as np
from numpy.lib.npyio import savetxt
import scipy.io
from numpy.core.numeric import NaN

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import matplotlib.animation

from numpy import ones,vstack
from numpy.linalg import lstsq

data_array = np.load('octoberfifteen.npy')

end = time.time()

#print("\nData summary:\n", data_array)
print("\nData shape:\n", data_array.shape)

f = []



data_array2 = data_array[:,1:40,:]

def average_points(a):
    avgs = []
    i = 0 
    while i < 3:
        t = 0 
        bad = 0 
        for point in a:
            if np.isnan(point[i]):
                bad = bad + 1
            else:
                t = t + point[i]
       
        if len(a) == bad:
            t = 0
        else:
            
            t = t/(len(a) - bad)
            
        avgs.append(t)
        i = i + 1

    return avgs

total_array = map(lambda x: average_points(x), data_array2)

#print(total_array)
result2 = np.array(total_array)
print("Data shape:", result2.shape)
#print(data_array2)
print("Data shape2:", data_array2.shape)

new_col = np.sum(result2,1).reshape((result2.shape[0],1))
np.append(result2,new_col,1)

result3 = np.array([0  ,0, 0])
print(result3)
i = 1

plt.hold(True)
# plt.xlim([-600,0])
# plt.ylim([600,2500])
i = 0


def count_nans(nans, one):
    
    count = 0 
    z = -3
    for p in range(6):
        # print("count", count)
        # print("nans: ", nans)
        # print("i", i)
        # print("i+z", (i+z))
        if np.isnan(nans):
            count = count + 1
        nans = data_array2[(i+z), one]
        
        z = z+1
    # print("count", count)
    return count




def replace_nans(cord, first, j):
    if np.isnan(cord):
        total_nans = count_nans(cord, first)
        if (total_nans > 4):
            cord = 0
            pass
        else:
            
            new_cord = 0
            for p in range(-3,4):
                data_array2[(j), first] = data_array2[(j+p), first]
                cord2 = data_array2[(j), first]
                if np.isnan(cord2):
                    pass
                else:
                    
                    new_cord = new_cord + cord2
                    #print("cord+cord2", new_cord)
                    cord = new_cord

                
            cord = cord / (6 - total_nans)
            #print("end of replace nans: ", cord)
            return cord

v = 0




def replace_nans2 (array, r, l, k):
    p = -3

    while np.isnan(array[l,r,k]) and p < 3 :
        array[l,r,k] = array[l+p,r,k]
        print("C", array[l,r,k])
        p = p+1


     


    #  while np.isnan(c):
    #         p = -3
    #         c = array[l + p, r, k]
    #         print("C", c)
    #         p = p+1
            
    #         if p > 6 :
    #             pass
    #         return c 

print ("first set:", data_array2[2000:2020,11,:])

data_array2 = data_array2[2000:2030,:,:]
index = 0
for row in data_array2:
    for i in range (35):
        X = row[i, 0]
        print("X", X)

        replace_nans2(data_array2,i, index, 0)
        replace_nans2(data_array2,i, index, 1)
        
        
        Y = row[i, 1]
        

    index = index + 1
    

print ("second set:", data_array2[2000:2020,11,:])            
    

# i = 0
# skips = 0
# for row in data_array2:
    
    
#     X2 = data_array2[i, 0, 0]
#     X2 = replace_nans(X2, 0,0, i)
#     print("X2", X2)

#     Y2 = data_array2[i, 0, 1]
#     Y2 = replace_nans(Y2, 0,1, i)
#     print("Y2", Y2)

#     X3 = data_array2[i, 1, 0]
#     X3 = replace_nans(X3, 1,0, i)
#     print("X3", X3)
#     Y3 = data_array2[i, 1, 1]
#     Y3 = replace_nans(Y3, 1,1, i)
#     print("Y3", Y3)

#     points = [(X2,Y2),(X3,Y3)]
#     x_coords, y_coords = zip(*points)
#     A = vstack([x_coords,ones(len(x_coords))]).T
#     m, c = lstsq(A, y_coords)[0]
#     print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
    


#     X = X2 - ((X2-X3) *.8) 
#     Y = Y2 - ((Y2-Y3) *.8)

    
#     print("X", X)
#     print("Y", Y)
    
#     print(" ")
   

#     plt.axes()
#     scatterplot1 = plt.plot(X2, Y2, 'bo')
#     scatterplot1 = plt.plot(X3, Y3, 'bo')
#     scatterplot1 = plt.plot(X, Y, 'ro')
    
    
    
#     if (X==0):
#         skips = skips+1
#         pass
#     else: 
#         plt.pause(.04)
    
    
  
#     if i < 213:
#         i = i + 1
#     print("i2", i)


# plt.show(scatterplot1)
# print('skips', skips)
# plt.close('all')




