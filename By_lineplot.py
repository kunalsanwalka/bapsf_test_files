# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:50:16 2020

@author: kunalsanwalka

This program plots the amplitude of the Alfven wave as a function of z
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

###############################################################################
#############################User defined variables############################
###############################################################################
freq=480 #KHz #Frequency of the antenna

#Data directory
data_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/Offset Antenna/'

#Savepath directory
savepath_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/By_axialAmp_offsetAntenna_freq_500KHz.png'
# #Name of the data files
# filenameArr=['By_XZ_Plane_freq_'+str(freq)+'KHz_col_500KHz.hdf','By_XZ_Plane_freq_'+str(freq)+'KHz_newBC.hdf','By_XZ_Plane_freq_'+str(freq)+'KHz_col_2000KHz.hdf','By_XZ_Plane_freq_'+str(freq)+'KHz_col_5000KHz.hdf']
# #Array with the legend for each datafile
# legendArr=[r'$\nu_{ei}$=500KHz',r'$\nu_{ei}$=1000KHz',r'$\nu_{ei}$=2000KHz',r'$\nu_{ei}$=5000KHz']

#Name of the data files
filenameArr=['By_XZ_Plane_freq_37KHz.hdf','By_XZ_Plane_freq_343KHz.hdf','By_XZ_Plane_freq_378KHz.hdf','By_XZ_Plane_freq_446KHz.hdf','By_XZ_Plane_freq_560KHz.hdf']
#Legend form each datafile
legendArr=['37','343','378','446','560']
###############################################################################

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

#Create the plot
fig=plt.figure(figsize=(10,4))
ax=fig.add_subplot(111)

#Plot the data from each file
for i in range(len(filenameArr)):
    #Define the full location of the data
    filepath=data_dir+filenameArr[i]

    #Get the data
    By,X,Z=dataArr(filepath)

    #Set time t to t=0
    By=np.real(By)

    #Interpolate the data to make plots cleaner
    ByGrid,xi,zi=interpData(By,X,Z)

    #Transpose to the get the amplitude along the z-line
    ByGrid=np.transpose(ByGrid)

    #Get the z-axis values
    zAxisVals=np.transpose(zi)[0]

    #Get the wave data 
    ByAxialAmp=ByGrid[1000]

    ax.plot(zAxisVals,ByAxialAmp,label=legendArr[i])

#Plot PML Line
ax.plot([8,8],[-1,1],label='PML Boundary',linestyle='dashed',color='k')

#Add labels
ax.set_title('Effect of PML Damping')
ax.set_xlabel('Z [m]')
ax.set_ylabel(r'$\Re(B_y)$')

#Change axis scaling and limits
# ax.set_yscale('symlog',linthreshy=10e-11)
ax.set_xlim(-6,8)
ax.set_ylim(-1.5e-6,1.5e-6)

#Define legend
ax.legend()
ax.legend(bbox_to_anchor=(1.205,1.03),loc='upper right')

#Miscellaneous
ax.grid(True)
ax.ticklabel_format(axis='y',style='sci',scilimits=(-1,1))

#Save the figure and close it
# plt.savefig(savepath_dir,dpi=600,bbox_inches='tight')
plt.show()
plt.close()