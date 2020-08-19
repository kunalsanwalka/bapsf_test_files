# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:31:27 2020

@author: kunalsanwalka
"""

import os
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
data_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/Ey_XY_Plane_freq_500KHz_fineMesh_12cm_col_5000KHz.hdf'

#Directory in which to save all the frames
savepath_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/EField_12cm/'

#Number of frames
numFrames=64

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

#Create the directory
if not os.path.exists(savepath_dir):
    os.makedirs(savepath_dir)

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

def createFrame(By,X,savepath_dir,frameNum,totFrames=numFrames):
    """
    This function takes the raw data and creates a contour plot for a specific frame

    Args:
        By: Array with the complex, frequency domain field value
        X: Array with the x-values
        savepath: Directory to save the frames
        frameNum: Which specific frame is being plotted (starts at 1)
        totFrames: Total number of frames in the animation
                 : Default value=numFrames
    """

    #Calcualate the time in the wave period
    time=(2*np.pi/omega)*((frameNum-1)/(totFrames-1))

    #Exponential to go from freq to time domain
    expVal=np.e**(-1j*omega*time)

    #Inverse fourier transform of By
    ByTD=By*expVal

    #Take the real value (as that is what we experimentally observe)
    ByReal=np.real(ByTD)
    
    #Create the plot
    fig=plt.figure(figsize=(10,8))
    plt.plot(X,ByReal)
    plt.grid(True)
    plt.xlim(-0.5,0.5)
    plt.ylim(-0.5,0.5)
    
    plt.title('12cm case')
    plt.xlabel('X [m]')
    plt.ylabel(r'$\Re(E_y)$')
    
    fig.savefig(savepath_dir+'frameNum_'+str(frameNum)+'.png',bbox_inches='tight',dpi=300)
    #plt.show()
    plt.close()

    return

def dataProcessing(filename):
    """
    This function outputs the fully porcessed data needed for the plotting
    function.
    
    Args:
        filename: Location of the file with the data to be processed
        
    Returns:
        EyArr: 1D Array of normalized Ey values along x for y=0
        xArr: Corresponding array with the x values
    """
    
    #Get the data
    Ey,X,Y=dataArr(data_dir)
    
    #Split data into real and imaginary parts
    EyReal=np.real(Ey)
    EyImag=np.imag(Ey)
    
    #Interpolate the data
    EyRealInterp,xi,yi=interpData(EyReal,X,Y)
    EyImagInterp,xi,yi=interpData(EyImag,X,Y)
    
    #Remove all the nan values
    EyRealInterp=np.nan_to_num(EyRealInterp)
    EyImagInterp=np.nan_to_num(EyImagInterp)
    
    #Combine the real and imaginary parts
    EyInterp=EyRealInterp+1j*EyImagInterp
    
    #Normalize the data
    maxVal=np.max(np.abs(EyInterp))
    EyInterp/=maxVal
    
    #Get the 1D array along x for y=0
    EyArr=EyInterp[1000]
    xArr=xi[1000]
    
    return EyArr,xArr

#%%
# =============================================================================
# Plotting Animations
# =============================================================================

#Get the data
EyArr,xArr=dataProcessing(data_dir)

#Create the animation frames
for i in tqdm(range(numFrames),position=0):
    createFrame(EyArr,xArr,savepath_dir,i)