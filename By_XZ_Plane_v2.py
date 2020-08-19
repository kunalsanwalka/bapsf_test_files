# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 11:58:00 2020

@author:kunalsanwalka

This program plots all the contours for an animation of By on the XZ Plane
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

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

#Antenna parameters
antOffset=-7 #m #Antenna offset from the center of the plasma

#Plotting parameters
numFrames=32 #Number of frames in the animation
#Data directory
data_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/Morales Solution/By_XZ_Plane_morales_withdamping.hdf'
#Directory in which to save all the frames
savepath_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/morales_withdamping_'+str(freq)+'KHz/'
#Directory to save the time averaged image
timeAvg_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/timeAvg_By_'+str(freq)+'KHz_fullChamber_col_'+str(col)+'KHz.png'

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

def createFrame(By,X,Z,savepath_dir,frameNum,totFrames=numFrames):
    """
    This function takes the raw data and creates a contour plot for a specific frame

    Args:
        By: Array with the complex, frequency domain field value
        X,Z: Co-ordinates of the z-value
        savepath: Directory to save the frames
        frameNum: Which specific frame is being plotted (starts at 1)
        totFrames: Total number of frames in the animation
                 : Default value=numFrames
    """
    
    #Normalized frequency
    freqNorm=np.round(omega/Omega_ne,2)

    #Calcualate the time in the wave period
    time=(2*np.pi/omega)*((frameNum-1)/(totFrames-1))

    #Exponential to go from freq to time domain
    expVal=np.e**(-1j*omega*time)

    #Inverse fourier transform of By
    ByTD=By*expVal

    #Take the real value (as that is what we experimentally observe)
    ByReal=np.real(ByTD)
    
    #Normalize the axes
    xNorm=X*Pi_e/c
    zNorm=(Z-antOffset)*Pi_e/c
    
    #cmap limit
    cmapLim=1e-6
    #Levels at which to draw contours
    numLevels=np.linspace(-cmapLim,cmapLim,100)
    
    #Create the plot
    fig=plt.figure(figsize=(16,8))
    plt.tricontourf(zNorm,xNorm,ByReal,numLevels,cmap='jet')
    plt.xlim(0,2500)
    plt.ylim(-60,60)
    
    plt.title(r'$\Re(B_y)$; $\omega/\Omega_{Ne}$='+str(freqNorm)+r'; $\nu_{ei}$='+str(col/1e6)+'MHz; Frame='+str(frameNum)+'/'+str(totFrames))
    plt.xlabel(r'$z\omega_{pe}/c$')
    plt.ylabel(r'$x\omega_{pe}/c$')
    
    fig.savefig(savepath_dir+'frameNum_'+str(frameNum)+'.png',bbox_inches='tight',dpi=300)
    #plt.show()
    plt.close()

    return

#%%
# =============================================================================
# Get the data from the file
# =============================================================================
    
By,X,Z=dataArr(data_dir)

print('Pulled data from the hdf5 file')

#%%
# =============================================================================
# Create the frames
# =============================================================================

for i in range(numFrames):
    print('Creating frame '+str(i+1)+' of '+str(numFrames))
    createFrame(By,X,Z,savepath_dir,frameNum=i+1)

#%%
# =============================================================================
# Create the time averaged plot
# =============================================================================

#Angular frequency
omega=2*np.pi*freq*1000

#Normalized frequency
freqNorm=np.round(freq/Omega_ne,2)

#Time step array
tArr=np.linspace(0,2*np.pi/omega,30)

#Exponential array to go to the time domain
expArr=np.e**(-1j*omega*tArr)

#Array to store the sum of different time slices
sumArr=np.array([0+0*1j]*len(By))
for i in range(len(expArr)):
    sumArr+=By*expArr[i]

#Only take the real part and average
ByAvg=np.abs(np.real(sumArr)/30)

#################################  Plotting  ##############################

#Increase the number of levels in the plot
maxExp=-6.0
minExp=-15.0
lev_exp=np.arange(minExp,maxExp+0.1,0.1)
numLevels=np.power(10,lev_exp)
#Array with the ticklabels
labelArr=[]
for i in range(int(minExp),int(maxExp)+1,1):
    labelArr.append(r'$10^{'+str(i)+'}$')
#Array with the values at which to place the ticks
tickArr=np.power(10,np.arange(minExp,maxExp+1,1))

#Plot the result
fig=plt.figure(figsize=(35,5))
ax=fig.add_subplot(111)
p=ax.tricontourf(Z,X,ByAvg,numLevels,locator=ticker.LogLocator())
ax.set_title(r'$\langle|\Re(B_y)|\rangle$; $\omega/Omega_{ci}$='+str(freqNorm)+r'; $\nu_{ei}$='+str(col)+'KHz')
ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]')
cbar=fig.colorbar(p,ticks=tickArr,pad=0.01)
cbar.ax.set_yticklabels(labelArr)
fig.savefig(timeAvg_dir,bbox_inches='tight',dpi=300)
plt.show()
plt.close()