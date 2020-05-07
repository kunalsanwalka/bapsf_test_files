# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:19:10 2020

@author: kunalsanwalka

This program plots the polarization of the Alfven wave as a function of z for
r=0.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

###############################################################################
#############################User defined variables############################
###############################################################################
freqArr=[37,80,108,206,275,343,378,446,480,500,530,560] #KHz #Frequency of the antenna

#Plasma Parameters
#Magnetic field
magB=0.15 #Tesla
#Density of the plasma
ne=0.5e18 #m^{-3}
#Collisionality of the plasma
nu_e=1e6 #Hz

#Data directory
data_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/New Density/'

#Savepath directory
savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/polarization_ratio_lineaverage.png'
###############################################################################

#Fundamental values
c=299792458             #Speed of light
eps_0=8.85418782e-12    #Vacuum Permittivity
q_e=-1.60217662e-19     #Electron Charge
q_p=1.60217662e-19      #Proton Charge
m_e=9.10938356e-31      #Electron Mass
m_amu=1.66053906660e-27 #Atomic Mass Unit

#Calculate the cyclotron frequency
#Helium cyclotron frequency
cycFreqHe=q_p*magB/(4*m_amu) #rad/s
#Neon cyclotron frequency
cycFreqNe=q_p*magB/(20*m_amu) #rad/s

#Normalize antenna frequency
normFreq=2*np.pi*np.array(freqArr)*1000/cycFreqNe

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
    
    #Remove the y component from the vertices
    xVals=[]
    yVals=[]
    newVertices=[]
    for vertex in vertices:
        xVals.append(vertex[0])
        yVals.append(vertex[2])
        newVertices.append([vertex[0],vertex[1]])
    vertices=newVertices
    
    #Convert to numpy arrays
    cdata=np.array(cdata)
    xVals=np.array(xVals)
    yVals=np.array(yVals)
    
    #Close the file
    f.close()
    
    return cdata, xVals, yVals

def interpData(ampData,xVals,zVals):
    """
    Interpolates the data into a regualar grid to allow for cleaner line plots

    Args:
        ampData: The raw data
        xVals,zVals: The co-ordinates of the data
    Returns:
        interpData: The interpolated data
        xi,zi: Meshgrid to allow for the plotting of the data
    """

    #Find the max and min of xVals and zVals to find the limits of the interpolation grid
    xmin=np.min(xVals)
    xmax=np.max(xVals)
    zmin=np.min(zVals)
    zmax=np.max(zVals)

    #Create the target grid of the interpolation
    xi=np.linspace(xmin,xmax,2001)
    zi=np.linspace(zmin,zmax,2001)
    xi,zi=np.meshgrid(xi,zi)

    #Interpolate the data
    interpData=griddata((xVals,zVals),ampData,(xi,zi),method='linear')

    return interpData,xi,zi

#Stores the lineaverage
lineavgArr=[]

#Plot the data from each file
for i in range(len(freqArr)):
    
    #Define the location of the data
    filepathX=data_dir+'Bx_XZ_Plane_freq_'+str(freqArr[i])+'KHz.hdf'
    filepathY=data_dir+'By_XZ_Plane_freq_'+str(freqArr[i])+'KHz.hdf'

    #Get the data
    Bx,X,Z=dataArr(filepathX)
    By,X,Z=dataArr(filepathY)

    #Left Handed Wave
    BLeft=(1/np.sqrt(2))*(Bx+1j*By)
    BLeftAbs=np.abs(BLeft)**2
    
    #Right Handed Wave
    BRight=(1/np.sqrt(2))*(Bx-1j*By)
    BRightAbs=np.abs(BRight)**2
    
    #Polarization Ratio
    polR=BLeftAbs/(BRightAbs+BLeftAbs)

    #Interpolate the data to make plots cleaner
    polRInterp,xi,zi=interpData(polR,X,Z)
    
    #Replace all nan values with 0
    polRInterp=np.nan_to_num(polRInterp)

    #Transpose to the get the amplitude along the z-line
    polRInterp=np.transpose(polRInterp)

    #Get the z-axis values
    zAxisVals=np.transpose(zi)[0]

    #Get the wave data 
    polRAxialAmp=polRInterp[1000]
    
    #Take the average of the polarization ratio
    lineavgArr.append(np.average(polRAxialAmp))
    
#Plot the line average data
plt.plot(normFreq,lineavgArr)

plt.xlabel(r'Normalized Angular Frequency [$\omega/\Omega_{Neon}$]')
plt.ylabel('LH Fraction [$|B_L|^2/(|B_R|^2+|B_L|^2)$]',rotation=90)

plt.grid(True)
plt.savefig(savepath,dpi=600,bbox_to_inches='tight')
plt.show()
plt.close()