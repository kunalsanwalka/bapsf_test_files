# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:50:16 2019

@author: kunalsanwalka

This program contains the functions to calculate the density changes based on the strength of the magnetic field
"""

import matplotlib.pyplot as plt

#Equilibrium Values
fwhm=0.3 #FWHM
bMag0=0.05 #Magnetic Field Strength
a=15 #Steepness of the dropoff

def Bfield(x,y,z):
    return 0,0,0

def magneticField(x,y,z):
    import numpy as np
    """
    Calculates the magnetic field vector at a given point
    
    Args:
        x,y,z: Position in the plasma (float)
    Returns:
        B: Magnetic Field Vector (array)
    """
    B=[0,0,0]
    
    #Scratch field
    #Goes from 500G to 1600G over 2m
    B[2]=(0.11/2)*(np.tanh(z)+1)+0.05
    
    return B

def Bnorm(x,y,z,forDens=False):
    import numpy as np
    """
    Calculates the magnitude of the magnetic field at a given point
    
    Args:
        x,y,z: Position in the plasma (float)
        forDens: If this function is being used for the density() function (boolean)
    Returns:
        Magnitude of the magnetic field (B) vector
    """
    if forDens==True:
        Bx,By,Bz=magneticField(x,y,z)
        return np.sqrt(Bx**2+By**2+Bz**2)
    else:
        Bx,By,Bz=Bfield(x,y,z)
        return np.sqrt(Bx**2+By**2+Bz**2)

def density(x,y,z):
    import numpy as np
    """
    Calculates the density of the plasma at a given point
    
    Args:
        x,y,z: Position at which we need to calculate the density (float)
    Returns:
        dens: Density of the plasma at a given (x,y,z) (float)
    """
    #Base Case
    dens=0
    
    #Base Density
    densBase=1e18
    
    #Magnetic Field magnitude
    bMag=Bnorm(x,y,z,True)

    #Radial Distance
    r=np.sqrt(x**2+y**2)
    
    #Mirror ratio
    mRatio=np.sqrt(bMag/bMag0)
    
    #Density
    dens=densBase*(np.tanh(-a*(mRatio*r-fwhm))+1)/2
    
    #Rescale density so that we maintain the same number of particles
    #Based on an analytical integral of the radial density function
    #Integration limit
    intLim=10
    #Numerator
    coshVal=np.cosh(a*(fwhm+intLim/mRatio))
    sechVal=1/np.cosh(a*(fwhm-intLim/mRatio))
    num=np.log(coshVal*sechVal)
    #Denominator
    coshVal=np.cosh(a*(fwhm+intLim))
    sechVal=1/np.cosh(a*(fwhm-intLim))
    den=np.log(coshVal*sechVal)
    #Rescale
    rescaleFac=num/(den*mRatio)
    dens/=rescaleFac
    
    return dens

import numpy as np

#Plot the density function
#Range over which to plot the density function
xvals=np.arange(-0.5,0.51,0.01)
yvals=np.arange(-0.5,0.51,0.01)
zvals=np.arange(-10,10.1,0.1)
#Number of levels in the contour plot
numLevels=100

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
#p4=plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/dens_contour.png',dpi=900,bbox_inches='tight')
p3=plt.show()

#Density function across r
label1='B='+str(np.round(magneticField(0,0,zvals[25])[2],2))+'T, z='+str(np.round(zvals[25],2))+'m'
label2='B='+str(np.round(magneticField(0,0,zvals[95])[2],2))+'T, z='+str(np.round(zvals[95],2))+'m'
label3='B='+str(np.round(magneticField(0,0,zvals[200])[2],2))+'T, z='+str(np.round(zvals[200],2))+'m'
p4=plt.plot(xvals,den[25],label=label1)
p4=plt.plot(xvals,den[95],label=label2)
p4=plt.plot(xvals,den[200],label=label3)
p4=plt.grid()
p4=plt.legend()
p4=plt.xlabel('r [m]')
p4=plt.ylabel('Density [$m^{-3}$]')
p4=plt.title('Effect of changing B field strength on the radial density function',y=1.05)
#p4=plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/dens_func.png',dpi=900,bbox_inches='tight')
p4=plt.show()

#Check to see if the number of particles is conserved
print('Number of particles at B='+str(np.round(magneticField(0,0,zvals[25])[2],2))+'T is- '+str(np.trapz(den[25],dx=0.01)))
print('Number of particles at B='+str(np.round(magneticField(0,0,zvals[95])[2],2))+'T is- '+str(np.trapz(den[95],dx=0.01)))
print('Number of particles at B='+str(np.round(magneticField(0,0,zvals[200])[2],2))+'T is- '+str(np.trapz(den[200],dx=0.01)))

#Density derivative
p5=plt.plot(xvals[1:],np.diff(den[25])/0.01,label=label1)
p5=plt.plot(xvals[1:],np.diff(den[95])/0.01,label=label2)
p5=plt.plot(xvals[1:],np.diff(den[200])/0.01,label=label3)
p5=plt.grid()
p5=plt.legend()
p5=plt.xlabel('r [m]')
p5=plt.ylabel('Slope of the density')
p5=plt.title('Derivative of the radial density function for different magnetic field strengths',y=1.05)
#p5=plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/Rhodot.png',dpi=900,bbox_inches='tight')
p5=plt.show()

#Magnetic field variation
fieldvals=[]
for i in zvals:
    B=magneticField(0,0,i)
    fieldvals.append(B[2])
    
p6=plt.plot(zvals,fieldvals)
p6=plt.grid()
p6=plt.xlabel('Z [m]')
p6=plt.ylabel('B [T]')
p6=plt.title('Magnetic field strength variation along the plasma chamber')
#p6=plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/Varying_B.png',dpi=900,bbox_inches='tight')
p6=plt.ylim(0,0.2)