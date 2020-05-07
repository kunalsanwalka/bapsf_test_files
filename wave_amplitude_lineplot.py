"""
@author: kunalsanwalka

This program plots the wave amplitude as a function of the z position
"""

###############################################################################
#############################User defined variables############################
###############################################################################
#Names of the read/write files
dataDir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/'
savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/wave_amplitude.png'
###############################################################################

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import ma
from scipy.interpolate import griddata
from matplotlib import ticker, cm

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
    xi=np.linspace(-0.5,0.5,1001)
    yi=np.linspace(-4,1.5,1001)
    xi,yi=np.meshgrid(xi,yi)

    #Interpolate the data
    interpData=griddata((xVals,yVals),ampData,(xi,yi),method='linear')

    return interpData,xi,yi

def linedata(filename):
    """
    This function takes in the filename and gives the wave power along the z-axis

    Args:
        filename: The path where the data is stored
    Returns:
        BArr: The wave power as a function of z
        zArr: The corresponding array with z-axis values
    """

    #Get the data
    By,X,Z=dataArr(filename)

    #Take the absolute value squared
    BAbs=np.abs(By)**2

    #Take the real data
    BAbs=np.real(By)

    #Get the interpolated data
    ByInterp,xi,zi=interpData(BAbs,X,Z)

    #Transpose so getting slices across r is easier
    BTrans=np.transpose(ByInterp)
    xi=np.transpose(xi)
    zi=np.transpose(zi)

    #Integrate along the z axis
    BArr=np.trapz(BTrans,axis=0)

    #Get the z-axis values
    zArr=zi[0]

    return BArr,zArr

#Create an array with the filenames
localNameArr=['By_XZ_Plane_freq_100KHz_col_0000KHz.hdf','By_XZ_Plane_freq_100KHz_col_2500KHz.hdf','By_XZ_Plane_freq_100KHz_col_5000KHz.hdf']
#Create an array with the labels for each lineplot
labelArr=['0MHz','2.5MHz','5MHz']

#Plot the data
fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('Effect of damping on wave amplitude')

for i in range(len(localNameArr)):
    BArr,zArr=linedata(dataDir+localNameArr[i])
    ax.plot(zArr,BArr,label=labelArr[i])

ax.set_yscale('symlog',linthreshy=1e-10)
ax.set_xlabel('Z [m]')
#ax.set_xlim(-1.5,1,5)
ax.set_ylabel('Wave Amplitude')
ax.legend()
ax.grid(True)
#plt.savefig(savepath,dpi=600)
plt.show()
plt.close()