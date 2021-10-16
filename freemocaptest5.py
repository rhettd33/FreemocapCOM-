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

plt.close('all')
data_array = np.load('StandingTestData.npy')

end = time.time()

#print("\nData summary:\n", data_array)
print("\nData shape:\n", data_array.shape)



# plt.xlim([-600,0])
# plt.ylim([600,2500])
i = 0
skips = 0
for i in range(213):
    
    for g in range(10,17):
        
        data_array2 = data_array[:,g:(g+2),:]

        def count_nans(nans, one, two):
        
            count = 0 
            z = -3
            for p in range(6):
                print("count", count)
                print("nans: ", nans)
                print("i", i)
                print("i+z", (i+z))
                if np.isnan(nans):
                    count = count + 1
                nans = data_array2[(i+z), one, two]
                
                z = z+1
            print("count", count)
            return count



        def replace_nans(cord, first, second, j):
            if np.isnan(cord):
                total_nans = count_nans(cord, first, second)
                if (total_nans > 4):
                    cord = 0
                    pass
                else:
                    z = -3
                    new_cord = 0
                    for p in range(6):
                        cord2 = data_array2[(j+z), first, second]
                        if np.isnan(cord2):
                            pass
                        else:
                            
                            new_cord = new_cord + cord2
                            print("cord+cord2", new_cord)
                            cord = new_cord

                        z = z+1
                    cord = cord / (6 - total_nans)
                    print("end of replace nans: ", cord)
            


                        
            return cord




            
            
        X2 = data_array2[i, 0, 0]
        X2 = replace_nans(X2, 0,0, i)
        print("X2", X2)

        Y2 = data_array2[i, 0, 1]
        Y2 = replace_nans(Y2, 0,1, i)
        print("Y2", Y2)

        X3 = data_array2[i, 1, 0]
        X3 = replace_nans(X3, 1,0, i)
        print("X3", X3)
        Y3 = data_array2[i, 1, 1]
        Y3 = replace_nans(Y3, 1,1, i)
        print("Y3", Y3)

        points = [(X2,Y2),(X3,Y3)]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        print("Line Solution is y = {m}x + {c}".format(m=m,c=c))



        X = X2 - ((X2-X3) *.8) 
        Y = Y2 - ((Y2-Y3) *.8)


        print("X", X)
        print("Y", Y)

        print(" ")


        plt.axes()
        
        plt.xlim(-500,500)
        plt.ylim(-500,500)

        scatterplot1 = plt.plot(X2, Y2, 'bo')
        scatterplot1 = plt.plot(X3, Y3, 'bo')
        scatterplot1 = plt.plot(X, Y, 'ro')
        
    
        
#plt.show(scatterplot1)

plt.close('all')










print('skips', skips)





