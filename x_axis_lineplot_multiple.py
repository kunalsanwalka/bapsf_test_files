# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 11:15:36 2020

@author: kunalsanwalka

This program plots the XY lineplots of multiple hdf files
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import griddata

#Fundamental values (S.I. Units)
c=299792458             #Speed of light
eps_0=8.85418782e-12    #Vacuum Permittivity
mu_0=1.25663706212e-6   #Vacuum permeability
q_e=-1.60217662e-19     #Electron Charge
q_p=1.60217662e-19      #Proton Charge
m_e=9.10938356e-31      #Electron Mass
m_amu=1.66053906660e-27 #Atomic Mass Unit

#Plasma parameters
freq=192 #KHz #Antenna frequency
B=0.1    #T #Background magnetic field strength
n=1.4e18 #m^{-3} #Density
heRatio=1.0 #Ratio of Helium in the plasma
neRatio=0.0 #Ratio of the Neon in the plasma
col=5*1e6  #Hz #Collisional damping rate

#Data directory
data_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/Morales Solution/'
#Image location
savepath_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/By_meshComparison.png'
#Data filenames
dataList=['By_XY_Plane_plasmaLen_350cm_noRefinement_z_250cm.hdf','By_XY_Plane_plasmaLen_350cm_narrowFineMesh_z_250cm.hdf','By_XY_Plane_plasmaLen_350cm_broadFineMesh_z_250cm.hdf','By_XY_Plane_plasmaLen_350cm_coarseMesh_z_250cm.hdf','By_XY_Plane_plasmaLen_350cm_broadCoarseMesh_z_250cm.hdf']
#Legend Array
legendArr=['No Refinement','Fine Mesh=10cm','Fine Mesh=12cm','Coarse Mesh=10cm','Coarse Mesh=20cm']
#Phase Array (as a fraction of 2pi)
phase=[0.5,0.78,0.78,0.9,0.9]

# =============================================================================
# Derived parameters
# =============================================================================
#Frequencies
omega=2*np.pi*freq*1000 #rad/s #Angular antenna frequency
Pi_he=np.sqrt(n*heRatio*q_p**2/(eps_0*4*m_amu)) #rad/s #Helium plasma frequency
Pi_ne=np.sqrt(n*neRatio*q_p**2/(eps_0*20*m_amu)) #rad/s #Neon plasma frequency
Pi_e=np.sqrt(n*q_e**2/(eps_0*m_e)) #rad/s #Electron plasma frequency
Omega_he=q_p*B/(4*m_amu) #rad/s #Helium cyclotron frequency
Omega_ne=q_p*B/(20*m_amu) #rad/s #Neon cyclotron frequency

#%%
# =============================================================================
# Functions
# =============================================================================

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

def dataProcessing(filename):
    """
    This function outputs the fully processed data needed for the plotting
    function.
    
    Args:
        filename: Location of the file with the data to be processed
        
    Returns:
        compArr: 1D Array of normalized component values along x for y=0
        xArr: Corresponding array with the x values
    """
    
    #Get the data
    comp,X,Y=dataArr(filename)
    
    #Split data into real and imaginary parts
    compReal=np.real(comp)
    compImag=np.imag(comp)
    
    #Interpolate the data
    compRealInterp,xi,yi=interpData(compReal,X,Y)
    compImagInterp,xi,yi=interpData(compImag,X,Y)
    
    #Remove all the nan values
    compRealInterp=np.nan_to_num(compRealInterp)
    compImagInterp=np.nan_to_num(compImagInterp)
    
    #Combine the real and imaginary parts
    compInterp=compRealInterp+1j*compImagInterp
    
    #Normalize the data
    maxVal=np.max(np.abs(compInterp))
    compInterp/=maxVal
    
    #Get the 1D array along x for y=0
    compArr=compInterp[1000]
    xArr=xi[1000]
    
    return compArr,xArr

#%%
# =============================================================================
# Plotting and data processing
# =============================================================================

p1=plt.figure(figsize=(15,12))

for i in range(len(dataList)):
    
    #Get the data
    compArr,xArr=dataProcessing(data_dir+dataList[i])
    
    #Get the absolute value
    absArr=np.abs(compArr)
    
    #Change the phase
    compArr*=np.e**(-1j*2*np.pi*phase[i])
    
    #Plot the data
    p1=plt.plot(xArr,absArr,label=legendArr[i])
    
p1=plt.title('Domain Transition Comparison')
p1=plt.xlabel('X [m]')
p1=plt.ylabel(r'$|B_y|$')
p1=plt.xlim(-0.5,0.5)
p1=plt.ylim(0,1)
p1=plt.grid()
p1=plt.legend()
p1=plt.show()