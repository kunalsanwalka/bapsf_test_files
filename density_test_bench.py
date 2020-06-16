# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:42:09 2019

@author: kunalsanwalka

This program calculates the density of the plasma as a function of the position in the chamber

##########
Verified
##########

"""

import matplotlib.pyplot as plt
import numpy as np

#Equilibrium Values
fwhm=0.5 #FWHM
bMag0=0.05 #Magnetic Field Strength
a=15 #Steepness of the dropoff

def density(x, y, z):
    
    #Base Case
    dens=0
    
    #Base Density
    densBase=1e18
    
    #Radial Distance
    r=np.sqrt(x**2+y**2)
    
    #Density
    dens=densBase*(np.tanh(-a*(r-fwhm))+1)/2 
    
    return dens
    
#Plot the density function

xvals=np.arange(-0.5,0.51,0.01)
yvals=np.arange(-0.5,0.51,0.01)
zvals=np.arange(-10,10.1,0.1)

#%%
# =============================================================================
# X lineplot
# =============================================================================

#Array to store the density
densArr=[]
for x in xvals:
    densArr.append(density(x,0,0))

#Plot the data
plt.plot(xvals,densArr)
plt.grid()
plt.xlim(-0.5,0.5)
plt.show()

#%%
########################################
#XY Plane#
########################################

#Array to store density
den=np.zeros(shape=(len(xvals),len(yvals)))

xcounter=0
ycounter=0
for i in xvals:
    for j in yvals:
        den[xcounter][ycounter]=density(i,j,zvals[50])
        ycounter+=1
    ycounter=0
    xcounter+=1

#Plot density across the cmap
X,Y=np.meshgrid(xvals,yvals)
numLevels=100
p1=plt.figure(figsize=(8,8))
p1=plt.contourf(X,Y,den,numLevels)
p1=plt.colorbar()
p1=plt.title('XY Plane, Z=-5m')
p1=plt.xlabel('X [m]')
p1=plt.ylabel('Y [m]')
#p1=plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/XY_Contour_Plot.png',dpi=900)
p1=plt.show()

#Plot the Y density function
p2=plt.figure(figsize=(8,4))
p2=plt.plot(yvals,den[50],label='Z=-5m')
p2=plt.xlim(-0.6,0.6)
p2=plt.title('Density along Y-Axis')
p2=plt.xlabel('Y [m]')
p2=plt.ylabel('Density [$m^{-3}$]')
p2=plt.grid()
#p2=plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/Y_Line_Plot.png',dpi=900)
p2=plt.show()

#%%
########################################
#XZ Plane#
########################################

#Array to store density
den=np.zeros(shape=(len(zvals),len(xvals)))

xcounter=0
zcounter=0
for i in zvals:
    for j in xvals:
        den[zcounter][xcounter]=density(j,0,i)
        xcounter+=1
    xcounter=0
    zcounter+=1
    
#Plot density across the cmap
X,Z=np.meshgrid(xvals,zvals)
p3=plt.figure(figsize=(8,8))
p3=plt.contourf(X,Z,den,numLevels)
p3=plt.colorbar()
p3=plt.title('XZ Plane')
p3=plt.xlabel('X [m]')
p3=plt.ylabel('Z [m]')
#p3=plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/XZ_Contour_Plot.png',dpi=900)
p3=plt.show()

#Plot the Z density function
#Create the array to be plotted
zden=[]
for i in range(len(zvals)):
    zden.append(den[i][51])
p5=plt.figure(figsize=(8,4))
p5=plt.plot(zvals,zden)
p5=plt.title('Density along Z-Axis')
p5=plt.xlabel('Z [m]')
p5=plt.ylabel('Density [$m^{-3}$]')
p5=plt.grid()
#p5=plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/Z_Line_Plot.png',dpi=900)
p5=plt.show()


#Plot the X density function
p4=plt.figure(figsize=(8,4))
p4=plt.plot(xvals,den[50],label='Z=-5m')
p4=plt.plot(xvals,den[150],label='Z=5m')
p4=plt.xlim(-0.6,0.6)
p4=plt.title('Density along X-Axis')
p4=plt.xlabel('X [m]')
p4=plt.ylabel('Density [$m^{-3}$]')
p4=plt.grid()
p4=plt.legend()
#p4=plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/X_Line_Plot.png',dpi=900)
p4=plt.show()