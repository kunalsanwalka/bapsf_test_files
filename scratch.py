# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:33:08 2020

@author: kunalsanwalka
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import griddata

###############################################################################
###########################  User defined variables  ##########################
###############################################################################

#Simulation frequency
#Note: This is the only variable defined with a linear frequency.
#      Every other frequency variable uses angular frequency.
freq=500

#Radial limit (radius upto which we want to integrate the solution)
radLim=0.03

#Array with the positions of the slices
zPosArr=np.arange(-525,801,25) #cm

#Plasma Parameters
#Magnetic field
magB=0.15 #Tesla
#Density of the plasma
ne=0.5e18 #m^{-3}
#Collisionality of the plasma
nu_e=5e6 #Hz

#Data Directory
dataDir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/freq_'+str(freq)+'KHz_col_'+str(int(nu_e/1000))+'KHz/'
#Savepath location
savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/handedness_ratio_XY_integral_col_'+str(int(nu_e/1000))+'KHz_r_50cm.png'

###############################################################################

#Fundamental values
c=299792458             #Speed of light
eps_0=8.85418782e-12    #Vacuum Permittivity
q_e=-1.60217662e-19     #Electron Charge
q_p=1.60217662e-19      #Proton Charge
m_e=9.10938356e-31      #Electron Mass
m_amu=1.66053906660e-27 #Atomic Mass Unit

#%%
def dataArr(filename):
    """
    This function takes the name of an hdf5 file and returns the relevant data arrays
    
    Data in the hdf5 file is organized as-

    Group 1- page1
        There is no data in this group, we can ignore it
        
    Group 2- page1.axes1.solid1
        This group has the following subgroups-
        Group 1- cdata
            Contains the data for the slice
            cdata has 2 subsequent subgroups-
                Group 1- Imag
                    Stores the complex part of the number
                Group 2- Real
                    Stores the real part of the number
        Group 2- idxset
            Do not know what data is stored here
        Group 3- vertices
            Contains the co-ordinates for the slice
    
    Args:
        filename: Name of the file (str)
    Returns:
        cdata: Complex valued solution of the problem (np.array)
        xVals: x-coordinate of the data points (np.array)
        yVals: y-coordinate of the data points (np.array)
    """
    #Open the file
    f=h5py.File(filename,'r')
    
    #Initialize the data arrays
    cdata=[]
    idxset=[]
    vertices=[]
    
    #Open groups in the file
    for group in f.keys():
#        print('Group- '+group)
        
        #Get the group
        currGroup=f[group]
        
        #Open keys in the group
        for key in currGroup.keys():
#            print('Key- '+key)
    
            #Append the data to the respective arrays
            if key=='cdata(Complex)':
                cdataGroup=currGroup[key]
                
                imag=[]
                real=[]
                #Open the keys in cdata
                for subkey in cdataGroup.keys():
#                    print('Subkey- '+subkey)
                    
                    #Get the real and imaginary parts of the array
                    if subkey=='Imag':
                        imag=cdataGroup[subkey][()]
                    elif subkey=='Real':
                        real=cdataGroup[subkey][()]
                
                #Convert lists to numpy arrays
                imag=np.array(imag)
                real=np.array(real)
                #Get the cdata value
                cdata=real+1j*imag
                
            elif key=='idxset':
                idxset=currGroup[key][()]
            elif key=='vertices':
                vertices=currGroup[key][()]
    
    #Remove the z component from the vertices
    xVals=[]
    yVals=[]
    newVertices=[]
    for vertex in vertices:
        xVals.append(vertex[0])
        yVals.append(vertex[1])
        newVertices.append([vertex[0],vertex[1]])
    vertices=newVertices
    
    #Convert to numpy arrays
    cdata=np.array(cdata)
    xVals=np.array(xVals)
    yVals=np.array(yVals)
    
    #Close the file
    f.close()
    
    return cdata, xVals, yVals

def interpData(ampData,xVals,yVals):
    """
    Interpolates the data into a regualar grid to allow for cleaner plots and simpler data manipulation

    Args:
        ampData: The raw data
        xVals,yVals: The co-ordinates of the data
    Returns:
        interpData: The interpolated data
        xi,yi: Meshgrid to allow for the plotting of the data
    """

    #Create the target grid of the interpolation
    xi=np.linspace(-radLim,radLim,101)
    yi=np.linspace(-radLim,radLim,101)
    xi,yi=np.meshgrid(xi,yi)

    #Interpolate the data
    interpData=griddata((xVals,yVals),ampData,(xi,yi),method='linear')

    return interpData,xi,yi

#%%
#Get the LH Fraction
ratioArr=[]
for z in zPosArr:
    
    #Location of the data
    BxString=dataDir+'Bx_XY_Plane_z_'+str(z)+'cm.hdf'
    ByString=dataDir+'By_XY_Plane_z_'+str(z)+'cm.hdf'
    
    #Get the data
    Bx,X,Y=dataArr(BxString)
    By,X,Y=dataArr(ByString)
    
    #Drop values beyond the radial limit
    #Array to traverse the list backwards
    indexArr=np.flip(np.arange(0,len(X),1))
    for i in indexArr:
        #Check if we are outside the radial limit
        if (X[i]**2+Y[i]**2)>radLim**2:
            #Remove elements out of bounds
            X=np.delete(X,i)
            Y=np.delete(Y,i)
            Bx=np.delete(Bx,i)
            By=np.delete(By,i)
            
    #Delete duplicate values
    #Construct the 2D numpy array
    temp2D=[]
    for i in range(len(X)):
        temp2D.append(np.array([X[i],Y[i],Bx[i],By[i]]))
    temp2D=np.array(temp2D)
    #Only keep the unique values
    unique2D=np.unique(temp2D,axis=0)
    #Deconstruct the 2D array into multiple 1D arrays
    X=[]
    Y=[]
    Bx=[]
    By=[]
    for i in range(len(unique2D)):
        X.append(unique2D[i][0])
        Y.append(unique2D[i][1])
        Bx.append(unique2D[i][2])
        By.append(unique2D[i][3])
    X=np.array(X)
    Y=np.array(Y)
    Bx=np.array(Bx)
    By=np.array(By)
    
    #Left Handed Wave
    BLeft=(1/np.sqrt(2))*(Bx+1j*By)
    BLeftAbs=np.abs(BLeft)**2
    #Right Handed Wave
    BRight=(1/np.sqrt(2))*(Bx-1j*By)
    BRightAbs=np.abs(BRight)**2
    
    #Interpolate the data
    BLeftInterpTemp,xi,yi=interpData(BLeftAbs,X,Y)
    BRightInterpTemp,xi,yi=interpData(BRightAbs,X,Y)

    #Replace all nan values with 0
    BLeftInterp=np.nan_to_num(BLeftInterpTemp)
    BRightInterp=np.nan_to_num(BRightInterpTemp)
    
    #Integrate over one axis
    tempIntegralLeft=np.trapz(BLeftInterp,axis=0)
    tempIntegralRight=np.trapz(BRightInterp,axis=0)
    #Integrate over the other axis
    integralLeft=np.trapz(tempIntegralLeft)
    integralRight=np.trapz(tempIntegralRight)

    #Get the LH Fraction
    ratio=integralLeft/(integralRight+integralLeft)
    
    #Append the fraction to the array
    ratioArr.append(ratio)