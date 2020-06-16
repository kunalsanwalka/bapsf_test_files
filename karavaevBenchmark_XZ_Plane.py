# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:50:11 2020

@author: kunalsanwalka
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':25}) #Change the fontsize
from scipy.interpolate import griddata

#Fundamental values (S.I. Units)
c=299792458             #Speed of light
eps_0=8.85418782e-12    #Vacuum Permittivity
q_e=-1.60217662e-19     #Electron Charge
q_p=1.60217662e-19      #Proton Charge
m_e=9.10938356e-31      #Electron Mass
m_amu=1.66053906660e-27 #Atomic Mass Unit

# =============================================================================
# User Defined Variables
# =============================================================================

#Plasma Parameters
freq=80     #KHz #Antenna Frequency
B=0.1       #T #Magnetic Field Strength
n=1.4e18    #m^{-3} #Density
col=4.75       #KHz #Collisionality

#Data directory
data_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/By_XZ_Plane_karavaev_withdamping_withPML_3rdOrder.hdf'
#Savepath Directory
savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/karavaev_withdamping_withPML_3rdOrder.png'

# =============================================================================
# Derived Variables
# =============================================================================

#Normalized Antenna Frequency
Omega_He=q_p*B/(4*m_amu) #Helium Cyclotron Frequency
normFreq=2*np.pi*1000*freq/Omega_He

#Electron plasma frequency
Pi_e=np.sqrt((n*q_e**2)/(eps_0*m_e))

#Plasma skin depth
plasmaSD=c/Pi_e
#%%
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
    Interpolates the data into a regualar grid to allow for cleaner vector plots

    Args:
        ampData: The raw data
        xVals,zVals: The co-ordinates of the data
    Returns:
        interpData: The interpolated data
        xi,zi: Meshgrid to allow for the plotting of the data
    """

    #Create the target grid of the interpolation
    xi=np.linspace(-0.5,0.5,1000)
    zi=np.linspace(-10,10,1000)
    xi,zi=np.meshgrid(xi,zi)

    #Decompose into real an imaginary parts
    realPart=np.real(ampData)
    imagPart=np.imag(ampData)

    #Interpolate the data
    realInterp=griddata((xVals,zVals),realPart,(xi,zi),method='linear')
    imagInterp=griddata((xVals,zVals),imagPart,(xi,zi),method='linear')

    #Reconstruct the data
    interpData=realInterp+1j*imagInterp

    return interpData,xi,zi

#%%
# =============================================================================
# Simulation data
# =============================================================================

#Pull data from the hdf5 file
By,X,Z=dataArr(data_dir)

#Interpolate the data
By,X,Z=interpData(By,X,Z)

#Normalize X and Z
X=X/plasmaSD
Z=Z/plasmaSD

#%%
# =============================================================================
# Plot the data
# =============================================================================

#Limits of the contour plot
contourMin=-9e-10
contourMax=9e-10

#Spacing of the contour lines
levelArr=np.linspace(contourMin,contourMax,500)

#Create the figure and add the data
p1=plt.figure(figsize=(35,5))
p1=plt.contourf(Z,X,np.real(By),levels=levelArr)

# #Colorbar
# cbar=plt.colorbar()
# cbar.set_label(r'$B_y$',rotation=270,labelpad=5)
# cbar=plt.clim(-1,1)

#Axis limits
p1=plt.xlim(0,2000)
p1=plt.ylim(-60,60)

#Axis labels
p1=plt.xlabel(r'$z / \delta_e$')
p1=plt.ylabel(r'$x / \delta_e$')
p1=plt.title(r'Gas= $He^+$;'+r' $\omega / \Omega_{He}=$'+str(np.round(normFreq,2))+r'; $\nu_{e}=$'+str(col)+r'KHz; $B_0=$'+str(B)+'T')

plt.grid(True)
plt.savefig(savepath,dpi=300,bbox_inches='tight')
plt.show()