# -*- coding: utf-8 -*-
"""
@author: kunalsanwalka

This program plots the left and right hand components of the waves on a contour plot
"""

###############################################################################
###########################  User defined variables  ##########################
###############################################################################
#Simulation variables
freq=200#KHz #Frequency of the antenna
Omega_ci=383.9#KHz #Ion cyclotron frequency
col=1000#KHz #Collisionallity of the plasma
z=0#m #Axial location of the data plane

#Names of the read/write files
filenameX='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/Bx_XY_Plane_freq_'+str(freq)+'KHz_highResCyl_20cm_col_'+str(col)+'KHz.hdf'
filenameY='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/By_XY_Plane_freq_'+str(freq)+'KHz_highResCyl_20cm_col_'+str(col)+'KHz.hdf'
savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/B_power_'+str(freq)+'KHz_highResCyl_20cm_col_'+str(col)+'KHz.png'

#Plotting options
xAxisLimits=[-0.5,0.5]
yAxisLimits=[-0.5,0.5]
###############################################################################

import h5py
import numpy as np
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

#Get the data
Bx,X,Y=dataArr(filenameX)
By,X,Y=dataArr(filenameY)

#Get the absolute values of the components
BxAbs=np.abs(Bx)
ByAbs=np.abs(By)

#Get the wavepower
BPower=np.sqrt(BxAbs**2+ByAbs**2)

#Normalize the wave power
BPowerNorm=BPower/np.max(BPower)

#Normalized antenna frequency
normFreq=np.round(freq/Omega_ci,2)

#################################  Plotting  #################################

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch

class HandlerEllipse(HandlerPatch):
    """
    This class allows for the legend to display ellipses as the icon for circles/ellipses rather than squares.

    Copied directly from matplotlib documentation under the 'Implementing a custom legend handler' subsection.

    Link to the documentation- https://matplotlib.org/tutorials/intermediate/legend_guide.html
    """
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=width + xdescent, height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

#Spacing of color levels
numLevels=np.linspace(0,1,100)

#Create the figure and add plot elements
fig=plt.figure(figsize=(9,8))
ax=fig.add_subplot(1,1,1)
p=ax.tricontourf(X,Y,BPowerNorm,levels=numLevels,vmax=1,vmin=0)
ax.set_title(r"Wave Power; $\omega/\Omega_{ci}$="+str(normFreq)+r"; $\nu_{ei}$="+str(col)+"KHz; Pure He")
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.grid(True)

#Add colorbar
fig.colorbar(p,ax=ax,ticks=np.linspace(0,1,11))

#Plot the mesh transition and density falloff boundaries
meshTransRadius=0.2
densFallRadius=0.3
meshTrans=plt.Circle((0,0),meshTransRadius,color='r',linewidth=2,fill=False)
densFall=plt.Circle((0,0),densFallRadius,color='orange',linewidth=2,fill=False)
ax.add_artist(meshTrans)
ax.add_artist(densFall)

#Add the radial spread markers
rSpread1=plt.Circle((0,0),0.03429951416966992,color='lime',linewidth=1,linestyle='dashdot',fill=False)
rSpread2=plt.Circle((0,0),0.07545893117327383,color='lime',linewidth=1,linestyle='dashdot',fill=False)
rSpread3=plt.Circle((0,0),0.11661834817687773,color='lime',linewidth=1,linestyle='dashdot',fill=False)
rSpread4=plt.Circle((0,0),0.18521737651621759,color='lime',linewidth=1,linestyle='dashdot',fill=False)
ax.add_artist(rSpread1)
ax.add_artist(rSpread2)
ax.add_artist(rSpread3)
ax.add_artist(rSpread4)

#Add the legend
lgd=fig.legend([meshTrans,densFall,rSpread1],['Mesh Transition','Density Falloff','Radial Spread'],handler_map={mpatches.Circle: HandlerEllipse()},bbox_to_anchor=(0.86,0.985),loc='upper right')

#Adjust zoom
ax.set_xlim(xAxisLimits)
ax.set_ylim(yAxisLimits)

#Save the plot
fig.savefig(savepath,bbox_extra_artists=(lgd,),bbox_to_anchor="tight",dpi=600)

plt.show()
plt.close()