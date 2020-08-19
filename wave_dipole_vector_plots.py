# -*- coding: utf-8 -*-
"""
Created on Fri Mar 6 13:26:30 2020

@author: kunalsanwalka

This program plots the vector components of the Alfven wave dipole and the 
associated current channels
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

###############################################################################
#############################User defined variables############################
###############################################################################
#User defined variables
freq=500#KHz #Frequency of the antenna
col=5000#KHz Collisionality of the plasma
z=1#m #Location of the place where data was taken

#Convert z to cm
zCm=z*100

#Number of frames in the animation
numFrames=60

#Names of the read/write files
#Data directory
data_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/'

#Filepath of the data
# filenameX=data_dir+'freq_'+str(freq)+'KHz_col_'+str(col)+'KHz/Bx_XY_Plane_z_'+str(int(np.round(zCm)))+'cm.hdf'
# filenameY=data_dir+'freq_'+str(freq)+'KHz_col_'+str(col)+'KHz/By_XY_Plane_z_'+str(int(np.round(zCm)))+'cm.hdf'
filenameX=data_dir+'Ex_XY_Plane_freq_500KHz_fineMesh_10cm_col_5000KHz.hdf'
filenameY=data_dir+'Ey_XY_Plane_freq_500KHz_fineMesh_10cm_col_5000KHz.hdf'

#Directory in which to save all the frames
savepath_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/EFIELD_dipole_'+str(freq)+'KHz_col_'+str(col)+'KHz_fineMesh_10cm/'

#Axes limits for the plots
xAxisLim=20
yAxisLim=20
###############################################################################

#Create the directory
if not os.path.exists(savepath_dir):
    os.makedirs(savepath_dir)

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
    Interpolates the data into a regualar grid to allow for cleaner vector plots

    Args:
        ampData: The raw data
        xVals,yVals: The co-ordinates of the data
    Returns:
        interpData: The interpolated data
        xi,yi: Meshgrid to allow for the plotting of the data
    """

    #Create the target grid of the interpolation
    xi=np.linspace(-0.5,0.5,101)
    yi=np.linspace(-0.5,0.5,101)
    xi,yi=np.meshgrid(xi,yi)

    #Decompose into real an imaginary parts
    realPart=np.real(ampData)
    imagPart=np.imag(ampData)

    #Interpolate the data
    realInterp=griddata((xVals,yVals),realPart,(xi,yi),method='linear')
    imagInterp=griddata((xVals,yVals),imagPart,(xi,yi),method='linear')

    #Reconstruct the data
    interpData=realInterp+1j*imagInterp

    return interpData,xi,yi

def newCurrent(BX,BY,xi,yi,expArr,t):
    """
    Calculates the current channels for the wave dipole
    
    Args:
        BX,BY: x and y components of the magnetic field (in frequency domain)
        xi, yi: Interpolated grids of the location of BX and BY
        expArr: Array to convert from the frequency to time domain
        t: Time in the waveperiod
    Returns:
        JzGrid: 2D Array with the values of the current
    """
    #Go from frequency to time domain
    BxTime=np.real(BX*expArr[t])
    ByTime=np.real(BY*expArr[t])

    #Find the xVals and yVals arrays
    xVals=xi[0]
    yVals=np.transpose(yi)[0]
    
    #Find dx and dy
    dx=xVals[1]-xVals[0]
    dy=yVals[1]-yVals[0]

    #Find the required derivatives
    dBXdyGrid,dBXdxGrid=np.gradient(BxTime,dx,dy)
    dBYdyGrid,dBYdxGrid=np.gradient(ByTime,dx,dy)

    #Find Jz
    JzGrid=dBYdxGrid-dBXdyGrid
    
    #Find divB
    # JzGrid=dBXdxGrid+dBYdyGrid

    return JzGrid

def plotData(BX,BY,xi,yi,expArr,t,savepath_dir):
    """
    Plots the data of the magnetic dipole and current channels

    Args:
        BXGrid,BYGrid: Bx and By values
        xi,yi: Grid points at which to plot Bx and By
        expArr: Array with the phase
        t: Phase of the data
        savepath_dir: Directory in which to save the image
    Returns:
        Saves the plot in a folder
    """
    
    #Find the current channel data
    Jz=newCurrent(BX,BY,xi,yi,expArr,t)

    #Find the dipole vector components
    BxTime=np.real(BX*expArr[t])
    ByTime=np.real(BY*expArr[t])

    #Plot the current density contour and dipole vector grid
    #Create the figure
    p1=plt.figure(figsize=(9,8))
    
    #Plot the data
    p1=plt.contourf(xi,yi,Jz,levels=100,vmin=-0.1,vmax=0.1)
    qv1=plt.quiver(xi,yi,BxTime,ByTime,width=0.004,scale=3)
    
    #Add axes labels and title
    p1=plt.xlabel('X [cm]',fontsize=20)
    p1=plt.ylabel('Y [cm]',fontsize=20)
    # p1=plt.title('Alfven Wave Dipole; Frequency='+str(freq)+r'KHz; $\nu_{ei}$='+str(col)+'KHz',fontsize=19,y=1.02)
    p1=plt.title('E Field; Frequency='+str(freq)+r'KHz; $\nu_{ei}$='+str(col)+'KHz',fontsize=19,y=1.02)
    
    #Set axes parameters
    p1=plt.xticks(np.arange(-50,51,5))
    p1=plt.yticks(np.arange(-50,51,5))
    p1=plt.xlim(-xAxisLim,xAxisLim)
    p1=plt.ylim(-yAxisLim,yAxisLim)
    
    #Add colorbar
    cbar=plt.colorbar()
    cbar.set_label('Normalized Current Density',rotation=270,labelpad=15)
    cbar=plt.clim(-1,1)
    
    #Add vector label
    plt.quiverkey(qv1,-0.1,-0.1,0.2,label=r'$(B_x,B_y)$')
    
    #Miscellaneous
    p1=plt.tick_params(axis='both', which='major', labelsize=18)
    p1=plt.grid(True)
    p1=plt.gcf().subplots_adjust(left=0.15)

    #Save the plot
    savepath_frame=savepath_dir+'frame'+str(t+1)+'.png'
    p1=plt.savefig(savepath_frame,dpi=100,bbox_to_anchor='tight')
    p1=plt.close()

    #Let me know which frame we just saved
    print('Saved frame '+str(t+1)+' of '+str(len(expArr)))
    
    return

#Create the array to go from the frequency to time domain
#Get the angular frequency
omega=2*np.pi*freq*1000
#Create the time array
tArr=np.linspace(0,2*np.pi/omega,numFrames)
#Create exponent array
expArr=np.e**(-1j*omega*tArr)

#Get the raw data
initU,Xu,Yu=dataArr(filenameX)
initV,Xv,Yv=dataArr(filenameY)
#Rescale
initU*=1e3 #Bx
initV*=1e3 #By

#Interpolate the data
BXGrid,xi,yi=interpData(initU,Xu,Yu)
BYGrid,xi,yi=interpData(initV,Xv,Yv)

#Convert the grid units to cm
xi*=100
yi*=100

#%%
# =============================================================================
# Generate all the frames for animation
# =============================================================================
for i in range(len(expArr)):
    plotData(BXGrid,BYGrid,xi,yi,expArr,i,savepath_dir)
    
#%%
# =============================================================================
# Linecut plot    
# =============================================================================

#Get the data along the y=0 axis
ByArr=BYGrid[50]
xArr=xi[50]

#Plot the data
plt.plot(xArr,ByArr)
plt.grid()
plt.xlabel('X [cm]')
plt.ylabel(r'$\Re(B_y)$')
plt.xlim(-40,40)
plt.show()
plt.close()