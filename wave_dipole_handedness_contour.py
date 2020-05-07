# -*- coding: utf-8 -*-
"""
@author: kunalsanwalka

This program plots the left and right hand components of the waves on a contour plot
"""

import h5py
import numpy as np
from scipy.interpolate import griddata

###############################################################################
###########################  User defined variables  ##########################
###############################################################################
#Simulation variables
#Note: The frequencies defined below are linear NOT ANGULAR
freq=480        #KHz #Frequency of the antenna
Omega_ci=115.17 #KHz #Neon cyclotron frequency
col=5000        #KHz #Collisionallity of the plasma
z=2.5           #m   #Axial location of the data plane

#Convert z to cm
zCm=z*100

#Names of the read/write files
filenameX='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/freq_'+str(freq)+'KHz_col_'+str(col)+'KHz/Bx_XY_Plane_z_'+str(int(np.round(zCm)))+'cm.hdf'
filenameY='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/freq_'+str(freq)+'KHz_col_'+str(col)+'KHz/By_XY_Plane_z_'+str(int(np.round(zCm)))+'cm.hdf'
savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/B_handedness_'+str(freq)+'KHz_col_'+str(col)+'KHz_z_'+str(int(np.round(zCm)))+'cm_r_12cm.png'

#Plotting options
xAxisLimits=[-0.12,0.12]
yAxisLimits=[-0.12,0.12]
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
    xi=np.linspace(-0.5,0.5,250)
    yi=np.linspace(-0.5,0.5,250)
    xi,yi=np.meshgrid(xi,yi)

    #Interpolate the data
    interpData=griddata((xVals,yVals),ampData,(xi,yi),method='linear')

    return interpData,xi,yi

#Get the data
Bx,X,Y=dataArr(filenameX)
By,X,Y=dataArr(filenameY)

#Left Handed Wave
BLeft=(1/np.sqrt(2))*(Bx+1j*By)
BLeftAbs=np.abs(BLeft)**2

#Right Handed Wave
BRight=(1/np.sqrt(2))*(Bx-1j*By)
BRightAbs=np.abs(BRight)**2

#Maximum value of the contour plot
maxVal=0
if np.max(BRightAbs)>np.max(BLeftAbs):
    maxVal=np.max(BRightAbs)
else:
    maxVal=np.max(BLeftAbs)

#Normalize the data with respect to the maximum value
BLeftAbs/=maxVal
BRightAbs/=maxVal

#Normalized antenna frequency
normFreq=np.round(freq/Omega_ci,2)

#################################  Plotting  ##################################

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch

plt.rcParams.update({'font.size':11})

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

def radialSpread(zVal):
    """
    This function calculates the radial spread values for the first 4 instances at which the wave crosses the data plane.
    Based on the z value of the data plane

    Args:
        z: The location at which the data was taken
    Returns:
        radArr: Array with the radial spread values of the first 4 wave crosses
    """

    #Lengths of the 3 sections of the plasma region
    lenArr=[1.5,zVal+2.5,1.5-zVal]

    #Distances the wave travels
    dArr=[lenArr[1],lenArr[1]+2*lenArr[2],2*lenArr[0]+lenArr[1],lenArr[0]+5.5+lenArr[2],2*lenArr[1]+lenArr[2]+5.5+lenArr[0]]
    dArr=np.array(dArr)

    #Create the terms for the radial spread formula
    massRatio=9.10938356e-31/(4*1.66053906660e-27) #Helium ions
    omegaBar=freq**2/Omega_ci**2
    freqTerm=1/(1-omegaBar**2)
    ratio=np.sqrt(massRatio*freqTerm)

    #Calculate the radial spreads
    radArr=ratio*dArr

    return radArr

#Create the subplots
fig=plt.figure(figsize=(13,5))
fig.suptitle(r"$\omega/\Omega_{Ne}$="+str(normFreq)+r"; $\nu_{ei}$="+str(col)+"KHz; Z="+str(np.round(zCm))+"cm; 50/50 He/Ne",fontsize=16,y=0.99,x=0.43)
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
#Make sure they have the same levels to allow for the same colorbar
numLevels=np.linspace(0,1,100)

#Plot the left handed wave
ax=ax1
p1=ax.tricontourf(X,Y,BLeftAbs,levels=numLevels,vmax=1,vmin=0)
ax.set_title('Left Handed Wave')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]',labelpad=-5)
ax.grid()
#Plot the right handed wave
ax=ax2
p2=ax.tricontourf(X,Y,BRightAbs,levels=numLevels,vmax=1,vmin=0)
ax.set_title('Right Handed Wave')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]',labelpad=-6)
ax.grid()

#Add the colorbar
cbar=fig.colorbar(p2,ax=[ax1,ax2],ticks=np.linspace(0,1,11))
cbar.set_label('Normalized Polarization',rotation=270,labelpad=15)

# #Plot the mesh transition and density falloff boundaries
# meshTransRadius=0.1
# densFallRadius=0.3
# #matplotlib does not allow us to use the same artist mutiple times on the same plot
# meshTrans=plt.Circle((0,0),meshTransRadius,color='r',linewidth=2,fill=False)
# meshTrans2=plt.Circle((0,0),meshTransRadius,color='r',linewidth=2,fill=False)
# densFall=plt.Circle((0,0),densFallRadius,color='orange',linewidth=2,fill=False)
# densFall2=plt.Circle((0,0),densFallRadius,color='orange',linewidth=2,fill=False)
# ax=ax1
# ax.add_artist(meshTrans)
# ax.add_artist(densFall)
# ax=ax2
# ax.add_artist(meshTrans2)
# ax.add_artist(densFall2)

# #Add the radial spread markers (matplotlib does not allow the same artist to be reused)
# radArr=radialSpread(z)
# #For left handed contour
# rSpread11=plt.Circle((0,0),radArr[0],color='lime',linewidth=1,linestyle='dashdot',fill=False)
# rSpread12=plt.Circle((0,0),radArr[1],color='lime',linewidth=1,linestyle='dashdot',fill=False)
# rSpread13=plt.Circle((0,0),radArr[2],color='lime',linewidth=1,linestyle='dashdot',fill=False)
# rSpread14=plt.Circle((0,0),radArr[3],color='lime',linewidth=1,linestyle='dashdot',fill=False)
# rSpread15=plt.Circle((0,0),radArr[4],color='lime',linewidth=1,linestyle='dashdot',fill=False)
# #For right handed contour
# rSpread21=plt.Circle((0,0),radArr[0],color='lime',linewidth=1,linestyle='dashdot',fill=False)
# rSpread22=plt.Circle((0,0),radArr[1],color='lime',linewidth=1,linestyle='dashdot',fill=False)
# rSpread23=plt.Circle((0,0),radArr[2],color='lime',linewidth=1,linestyle='dashdot',fill=False)
# rSpread24=plt.Circle((0,0),radArr[3],color='lime',linewidth=1,linestyle='dashdot',fill=False)
# rSpread25=plt.Circle((0,0),radArr[4],color='lime',linewidth=1,linestyle='dashdot',fill=False)
# ax=ax1
# ax.add_artist(rSpread11)
# ax.add_artist(rSpread12)
# ax.add_artist(rSpread13)
# ax.add_artist(rSpread14)
# ax.add_artist(rSpread15)
# ax=ax2
# ax.add_artist(rSpread21)
# ax.add_artist(rSpread22)
# ax.add_artist(rSpread23)
# ax.add_artist(rSpread24)
# ax.add_artist(rSpread25)

# #Add the legend
# lgd=fig.legend([meshTrans,densFall],['Mesh Transition','Density Falloff'],handler_map={mpatches.Circle: HandlerEllipse()},bbox_to_anchor=(1,0.91),loc='upper right')

#Adjust axis limits for zoom functionality
ax=ax1
ax.set_xlim(xAxisLimits)
ax.set_ylim(yAxisLimits)
ax=ax2
ax.set_xlim(xAxisLimits)
ax.set_ylim(yAxisLimits)

#Save the plot
# fig.savefig(savepath,bbox_extra_artists=(lgd,),dpi=600)
fig.savefig(savepath,bbox_inches='tight',dpi=600)

plt.show()
plt.close()