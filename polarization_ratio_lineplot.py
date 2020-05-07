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
savepath_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/Axial Polarization Ratio/'
###############################################################################

#Create the directory
if not os.path.exists(savepath_dir):
    os.makedirs(savepath_dir)

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

def radialAvg(data,xi,zi,rad):
    """
    This function returns the data along z for r=0 but avergaed around +-rad
    NOTE: Because data, xi and zi are from interpolation, they are square
          arrays.
    
    Args:
        data: Interpolated data grid
        xi,zi: Interpolated position grids
        rad: Radius over which to average
    Returns:
        avgData: data along z for r=0 averaged along +-rad
    """
    #Remove all nan values from data
    data=np.nan_to_num(data)
    
    #Array with x values
    xArr=xi[0]
    
    #Array with z values
    zArr=np.transpose(zi)[0]
    
    #Create the array to store the values
    avgData=[]
    
    #Go over each z position
    for i in range(len(zArr)):
        
        #Store the total at each z position
        zPosTot=0
        
        #Counter to help average
        count=0
        
        #Go over each x position
        for j in range(len(xArr)):
            
            #Check if we are within the radius
            if np.abs(xArr[j])<=rad:
                
                #Add the data to the position total
                zPosTot+=data[i][j]
                
                #Increment the counter
                count+=1
        
        #Calculate the radial average
        zPosAvg=zPosTot/count
        
        #Add to the array
        avgData.append(zPosAvg)
    
    return avgData

# #Create the plot
# fig=plt.figure(figsize=(15,4))
# ax=fig.add_subplot(111)

#Plot the data from each file
for i in range(len(freqArr)):
    
    #Define the location of the data
    filepathX=data_dir+'Bx_XZ_Plane_freq_'+str(freqArr[i])+'KHz.hdf'
    filepathY=data_dir+'By_XZ_Plane_freq_'+str(freqArr[i])+'KHz.hdf'
    
    #Define the legend
    legend=r'$\omega / \Omega_{Neon}$='+str(np.round(normFreq[i],2))

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

    #Get the z-axis values
    zAxisVals=np.transpose(zi)[0]

    #Get the wave data 
    polRAxialAmp=radialAvg(polRInterp,xi,zi,0.1)
    
# =============================================================================
#     Plotting
# =============================================================================
    
    #Create the plot
    fig=plt.figure(figsize=(15,4))
    ax=fig.add_subplot(111)

    #Plot the data
    ax.plot(zAxisVals,polRAxialAmp)#,label=legend)

    #Add labels
    ax.set_title('Polarization Ratio; '+legend)
    ax.set_xlabel('Z [m]')
    ax.set_ylabel(r'LH Fraction [$|B_L|^2/(|B_R|^2+|B_L|^2)$]',rotation=90)
    
    #Change axis scaling and limits
    ax.set_xlim(-10,10)
    ax.set_ylim(0,1)
    
    # #Define legend
    # ax.legend()
    # ax.legend(bbox_to_anchor=(1.25,1.03),loc='upper right')
    
    #Miscellaneous
    ax.grid(True)
    # ax.ticklabel_format(axis='y',style='sci',scilimits=(-1,1))
    
    #Define the savepath
    savepath=savepath_dir+str(freqArr[i])+'KHz_r_10cm.png'
    
    #Save the figure and close it
    plt.savefig(savepath,dpi=600,bbox_inches='tight')
    plt.show()
    plt.close()