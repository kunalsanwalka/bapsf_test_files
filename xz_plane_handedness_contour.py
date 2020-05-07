"""
@author: kunalsanwalka

This program plots the left and right hand components of the waves on a contour plot
"""

###############################################################################
#############################User defined variables############################
###############################################################################
#User defined variables
freq=200#KHz #Frequency of the antenna
Omega_ci=383.9#KHz #Ion cyclotron frequency
z=0#m #Axial location of the data plane

#Names of the read/write files
filenameX='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/Bx_XZ_Plane_freq_'+str(freq)+'KHz_fullChamber.hdf'
filenameY='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/By_XZ_Plane_freq_'+str(freq)+'KHz_fullChamber.hdf'
savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/B_handedness_'+str(freq)+'KHz_fullChamber.png'
###############################################################################

import h5py
import numpy as np
from numpy import ma
from scipy.interpolate import griddata

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
    xi=np.linspace(-0.5,0.5,51)
    yi=np.linspace(-4,1.5,51)
    xi,yi=np.meshgrid(xi,yi)

    #Interpolate the data
    interpData=griddata((xVals,yVals),ampData,(xi,yi),method='linear')

    return interpData,xi,yi

#Get the data
Bx,X,Z=dataArr(filenameX)
By,X,Z=dataArr(filenameY)

#Left Handed Wave
BLeft=(1/np.sqrt(2))*(Bx+1j*By)
BLeftAbs=np.abs(BLeft)**2

#Right Handed Wave
BRight=(1/np.sqrt(2))*(Bx-1j*By)
BRightAbs=np.abs(BRight)**2

#Remove the data close to the antenna
#for i in range(len(X)-1,-1,-1):
#    #Iterate backwards
#    if Z[i]<-3 or Z[i]>0.5:
#        Z=np.delete(Z,i)
#        X=np.delete(X,i)
#        BRightAbs=np.delete(BRightAbs,i)
#        BLeftAbs=np.delete(BLeftAbs,i)

#Normalize the data with respect to the local field
BLeftTemp=np.array(BLeftAbs,copy=True)
BRightTemp=np.array(BRightAbs,copy=True)
BLeftAbs/=(BLeftTemp+BRightTemp)
BRightAbs/=(BLeftTemp+BRightTemp)

#Normalized antenna frequency
normFreq=np.round(freq/Omega_ci,2)

##Maximum value of the contour plot
#maxVal=0
#if np.max(BRightAbs)>np.max(BLeftAbs):
#    maxVal=np.max(BRightAbs)
#else:
#    maxVal=np.max(BLeftAbs)

##Normalize the data with respect to the maximum value
#BLeftAbs/=maxVal
#BRightAbs/=maxVal

#################################  Plotting  #################################

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import ticker, cm

#Change the fontsize
plt.rcParams.update({'font.size': 18})

#Create the subplots
fig=plt.figure(figsize=(35,10))
fig.suptitle(r"($\omega/\Omega_{ci}$)="+str(normFreq)+"; Pure He",y=0.97,x=0.445)
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)

#Make sure they have the same levels to allow for the same colorbar
#lev_exp=np.arange(-12.0,1.0,0.25)
#numLevels=np.power(10,lev_exp)
numLevels=np.linspace(0,1,101)

#Plot the left handed wave
ax=ax1
p1=ax.tricontourf(Z,X,BLeftAbs,numLevels,cmap='coolwarm')#,locator=ticker.LogLocator())
#Plot the demarcation of the damping layer
#p1=ax.plot([-0.5,0.5],[-3,-3],color='red',linewidth=2)
#p1=ax.plot([-0.5,0.5],[0.5,0.5],color='red',linewidth=2)
ax.set_title('Left Handed Wave')
ax.set_ylabel('X [m]',labelpad=-5)
ax.grid()

#Plot the right handed wave
ax=ax2
p2=ax.tricontourf(Z,X,BRightAbs,numLevels,cmap='coolwarm')#,locator=ticker.LogLocator())
#Add the colorbar
cbar=fig.colorbar(p2,ax=[ax1,ax2],pad=0.015,ticks=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
cbar.ax.set_yticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
#Plot the demarcation of the damping layer
#p2=ax.plot([-0.5,0.5],[-3,-3],color='red',linewidth=2)
#p2=ax.plot([-0.5,0.5],[0.5,0.5],color='red',linewidth=2)
ax.set_title('Right Handed Wave')
ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]',labelpad=-5)
ax.grid()

#Save the plot
fig.savefig(savepath,dpi=450,bbox_inches='tight')

#plt.show()
plt.close()