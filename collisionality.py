"""
This program calculates the collisionality term as function of the position in the plasma
"""

def collisionality(x,y,z):
    """
    This function calculates the collisionality as a function of the position in the plasma

    Args:
        x,y,z: Position in the plasma
    Returns:
        col: Collisionality (float)
    """
    import numpy as np
    
    #Initialize col
    col=1*1e6 #MHz #Base case collisionality

    #col is uniform through most of the plasma
    if z<=9 and z>=-9:
        return col
    #Ramp the collisionality at the end to make sure all the wave energy is absorbed before it hits the walls
    elif z>9:
        #Quadratic ramp for the collisionality
        zLoc=z-9 #Work in local co-ordinates
        col+=1e8*(zLoc**2) #Ramps to 100MHz
    elif z<-9:
        #Quadratic ramp for the collisionality
        zLoc=-z-9 #Work in local co-ordinates
        col+=1e8*(zLoc**2) #Ramps to 100MHz
    return col

#Plot the function to make sure it is real
import numpy as np
import matplotlib.pyplot as plt

#Arrays to store the position and collisionality
colArr=[]
zArr=np.arange(-10,10,0.01)

#Calculate the collisionality
for i in range(len(zArr)):
    colArr.append(collisionality(0,0,zArr[i]))

plt.plot(zArr,colArr)
plt.grid(True)
plt.show()