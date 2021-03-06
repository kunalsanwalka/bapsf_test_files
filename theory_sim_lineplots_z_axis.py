# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:03:25 2020

@author: kunalsanwalka

This program creates a comparison plot between the Morales theory and the
Petra-M simulation.

It does this by creating a bunch of frames each with a slightly different phase
for the theory plot so you can line them up with the same phase for the ideal
comparison.
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
freq=500 #KHz #Antenna frequency
B=0.15    #T #Background magnetic field strength
n=1.4e18 #m^{-3} #Density
heRatio=1.0 #Ratio of Helium in the plasma
neRatio=0.0 #Ratio of the Neon in the plasma
colE=5*1e6  #Hz #Collisional damping rate

#Antenna parameters
antOffset=-7 #m #Distance of the antenna from the center of the chamber

#Data directory
data_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/Morales Solution/'
#Savepath directory
savepath_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/'

#Location of simulation data
sim_data_dir=data_dir+'By_XZ_Plane_freq_500KHz_fineMesh_12cm_col_5000KHz.hdf'
#Location of frames folder
savepath_frames=savepath_dir+'By_lineplot_frames_fineMesh_12cm/'

#Plotting parameters (distances are wrt the antenna)
nearFieldLim=1 #m #Distance of near field values being ignored
pmlLim=15 #m #Location at which the PML begins
numFrames=32 #Number of frames to choose from

# =============================================================================
# Derived parameters
# =============================================================================
#Frequencies
omega=2*np.pi*freq*1000 #rad/s #Angular antenna frequency
Pi_he=np.sqrt(n*q_p**2/(eps_0*4*m_amu)) #rad/s #Helium plasma frequency
Pi_e=np.sqrt(n*q_e**2/(eps_0*m_e)) #rad/s #Electron plasma frequency
Omega_he=q_p*B/(4*m_amu) #rad/s #Helium cyclotron frequency

#Create the directory
if not os.path.exists(savepath_frames):
    os.makedirs(savepath_frames)

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
    Interpolates the data into a regualar grid to allow for cleaner plots and simpler data manipulation

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

def createFrame(BySim,ByTheory,zSim,zTheory,savepath_frames,frameNum,totFrames=numFrames):
    """
    This function takes the raw data and creates a line plot for a specific frame

    Args:
        BySim,ByTheory: Arrays with the complex, frequency domain field value
        zSim,zTheory: z-axis values for By
        savepath_frames: Directory to save the frames
        frameNum: Which specific frame is being plotted (starts at 1)
        totFrames: Total number of frames in the animation
                 : Default value=numFrames
    """
    
    #Normalized frequency
    freqNorm=np.round(omega/Omega_he,2)

    #Calcualate the time in the wave period
    time=(2*np.pi/omega)*((frameNum-1)/(totFrames-1))

    #Exponential to go from freq to time domain
    expVal=np.e**(-1j*omega*time)

    #Inverse fourier transform of By
    ByTheoryTD=ByTheory*expVal

    #Take the real value (as that is what we experimentally observe)
    ByTheoryReal=np.real(ByTheoryTD)
    BySimReal=np.real(BySim)
    
    #Normalize the axes
    zTheoryNorm=zTheory*Pi_e/c
    zSimNorm=zSim*Pi_e/c
    
    #Normalize the simulations
    ByTheoryNorm=ByTheoryReal/abs(max(ByTheoryReal,key=abs))
    BySimNorm=BySimReal/abs(max(BySimReal,key=abs))
    
    #Create the plot
    fig=plt.figure(figsize=(16,8))
    plt.plot(zTheoryNorm,ByTheoryNorm,label='Theory')
    plt.plot(zSimNorm,BySimNorm,label='Simulation')
    plt.xlim(nearFieldLim*Pi_e/c,max(zSimNorm))
    plt.ylim(-1,1)
    
    plt.title(r'$\omega/\Omega_{He}$='+str(freqNorm)+r'; $\nu_{ei}$='+str(colE/1e6)+'MHz; Frame='+str(frameNum)+'/'+str(totFrames))
    plt.xlabel(r'$z\omega_{pe}/c$')
    plt.ylabel(r'$\Re(B_y)$')
    
    plt.legend()
    plt.grid()
    
    fig.savefig(savepath_frames+'frameNum_'+str(frameNum)+'.png',bbox_inches='tight',dpi=300)
    #plt.show()
    plt.close()

    return

#%%
# =============================================================================
# Get the data
# =============================================================================

#Get the theoretical data
ByTheory=np.load(data_dir+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_By_Z_lineplot.npy')
zArrTheory=np.load(data_dir+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_zArr_Z_lineplot.npy')

#Get the simulation data
BySim,xSim,zSim=dataArr(sim_data_dir)

#Interpolate the simulation data
BySim,xSim,zSim=interpData(BySim,xSim,zSim)

#Get the r=0 line for the simulation data
#Transpose to the get the amplitude along the z-line
BySim=np.transpose(BySim)[1000][1:-1]

#Get the z-axis values
zArrSim=np.transpose(zSim)[0][1:-1]-antOffset

#Remove near antenna and PML values
#For the theoretical data
for i in np.flip(range(len(zArrTheory))):
    if zArrTheory[i]<nearFieldLim or zArrTheory[i]>pmlLim:
        zArrTheory=np.delete(zArrTheory,i)
        ByTheory=np.delete(ByTheory,i)
#For the simulation data
for i in np.flip(range(len(zArrSim))):
    if zArrSim[i]<nearFieldLim or zArrSim[i]>pmlLim:
        zArrSim=np.delete(zArrSim,i)
        BySim=np.delete(BySim,i)

#%%
# =============================================================================
# Create the plot
# =============================================================================

for i in tqdm(range(numFrames),position=0):
    createFrame(BySim,ByTheory,zArrSim,zArrTheory,savepath_frames,i+1)