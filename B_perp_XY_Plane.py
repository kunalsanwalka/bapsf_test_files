# -*- coding: utf-8 -*-
"""
Created on Sun May 31 19:23:51 2020

@author: kunalsanwalka
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import griddata
plt.rcParams.update({'font.size':20})

#Fundamental values
c=299792458             #Speed of light
eps_0=8.85418782e-12    #Vacuum Permittivity
q_e=-1.60217662e-19     #Electron Charge
q_p=1.60217662e-19      #Proton Charge
m_e=9.10938356e-31      #Electron Mass
m_amu=1.66053906660e-27 #Atomic Mass Unit

# =============================================================================
# User defined variables
# =============================================================================

#Plasma Parameters
freq=80                 #KHz #Antenna Frequency
magB=0.1                #T      #Magnetic Field Stength
ne=1.3e18               #m^{-3} #Plasma Density
nu_e=1.75e7             #Hz     #Plasma Collisionality
heRatio=1.0             #arb.   #Helium Fraction
neRatio=0.0             #arb.   #Neon Fraction
lambdaPerp=0.44822261   #m      #Perpendicular wavelength

#Data Directory
data_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/Karavaev Benchmark/'

#Data Filenames
filenameX='Bx_XY_Plane_freq_80KHz_withDamping_withPML_3rdOrder.hdf'
filenameY='By_XY_Plane_freq_80KHz_withDamping_withPML_3rdOrder.hdf'

#Frame Directory
savepath_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/B_perp_karavaev_freq_80KHz_withDamping_withPML_3rdOrder/'

#Number of frames
numFrames=8

#Create the directory
if not os.path.exists(savepath_dir):
    os.makedirs(savepath_dir)

#%%

# =============================================================================
# Derived Values
# =============================================================================

Omega_He=q_p*magB/(4*m_amu) #Helium Cyclotron Frequency

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
    xi=np.linspace(-0.5,0.5,201)
    yi=np.linspace(-0.5,0.5,201)
    xi,yi=np.meshgrid(xi,yi)

    #Interpolate the data
    interpData=griddata((xVals,yVals),ampData,(xi,yi),method='linear')

    return interpData,xi,yi

def createFrame(Bx,By,X,Y,freq,savepath_dir,frameNum,totFrames=numFrames):
    """

    Args:
        Bx (numpy.array): B_x component of the data.
        By (numpy.array): B_y component of the data.
        X (numpy.array): x position of the data.
        Y (numpy.array): y position of the data.
        freq (int): Linear frequency of the antenna.
        savepath_dir (string): Directory in which to save all the frames.
        frameNum (int): Current frame number.
        totFrames (int, optional): Total number of frames in the animation. Defaults to numFrames.

    Returns:
        Saves the frame in the savepath_dir directory
        
    """
    
    #Angular frequency
    omega=2*np.pi*freq*1000
    
    #Normalized angular frequency
    freqNorm=np.round(omega/Omega_He,2)

    #Calcualate the time in the wave period
    time=(2*np.pi/omega)*((frameNum-1)/(totFrames-1))

    #Exponential to go from freq to time domain
    expVal=np.e**(-1j*omega*time)

    #Inverse fourier transform
    BxTD=Bx*expVal
    ByTD=By*expVal

    #Take the real value (as that is what we experimentally observe)
    BxReal=np.real(BxTD)
    ByReal=np.real(ByTD)
    
    #Get the perpendicular component
    B_perp=np.sqrt(BxReal**2+ByReal**2)
    
    #Interpolate the data
    BPerpInterp,xi,yi=interpData(B_perp,X,Y)
    
    # =============================================================================
    # Plot the data  
    # =============================================================================
    
    plt.figure(figsize=(8,8))
    
    #Plot the data
    plt.contourf(xi,yi,BPerpInterp,levels=500,vmin=0,vmax=1.5e-10,cmap='jet')
    
    #Add title and axes labels
    plt.xlabel('Y [m]')
    plt.ylabel('X [m]')
    
    #Add axes limits
    plt.xlim(-0.25,0.25)
    plt.ylim(-0.25,0.25)
    
    #Add the grid
    plt.grid(True)
    
    #Save the figure
    plt.savefig(savepath_dir+'frameNum_'+str(frameNum)+'.png',bbox_inches='tight',dpi=300)
    plt.close()

#%%
    
# =============================================================================
# Analysis and Ploting
# =============================================================================

#Get the data
Bx,X,Y=dataArr(data_dir+filenameX)
By,X,Y=dataArr(data_dir+filenameY)

plt.figure(figsize=(10,10))
plt.scatter(Y,X)
plt.xlim(-0.15,0.15)
plt.ylim(-0.15,0.15)
plt.show()
plt.close()

#Create the frames
for i in range(numFrames):
    print('Creating frame '+str(i+1)+' of '+str(numFrames))
    createFrame(Bx,By,X,Y,freq,savepath_dir,i+1)
