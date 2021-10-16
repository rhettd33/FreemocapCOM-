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



data_array = np.load('StandingTestData.npy')

end = time.time()

#print("\nData summary:\n", data_array)
print("\nData shape:\n", data_array.shape)

f = []



data_array2 = data_array[:,20:22,:]

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
for row in result2:
    
    
    X2 = data_array2[i, 0, 0]
    print("X2", X2)
    Y2 = data_array2[i, 0, 1]
    print("Y2", Y2)

    X3 = data_array2[i, 1, 0]
    print("X3", X3)
    Y3 = data_array2[i, 1, 1]
    print("Y3", Y3)
    


    X = (X2 + X3) / 2
    Y = (Y2 +Y3) / 2

    
    print("X", X)
    print("Y", Y)
    i = i+1
    print(" ")
   

    plt.axes()
    scatterplot1 = plt.plot(X2, Y2, 'bo')
    scatterplot1 = plt.plot(X3, Y3, 'bo')
    scatterplot1 = plt.plot(X, Y, 'ro')
    
    
    
    if np.isnan(X):
        pass
    else: 
        plt.pause(.5)
    i = i + 1


plt.show(scatterplot1)
plt.close('all')




