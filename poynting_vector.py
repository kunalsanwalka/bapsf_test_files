# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:32:37 2019

@author: kunalsanwalka

This program calculates the Poynting Vector flux through a given xy-plane
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

###############################################################################
#############################User defined variables############################
###############################################################################

#Frequency
freq=250#KHz #Frequency of the antenna
zArr=[6,0,-6]#m #z co-ordinate of the various data planes
savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/Pflux_'+str(freq)+'KHz.png'

###############################################################################

def filename(freq,z,comp):
    """
    This function creates the correct filename from which to fetch the data
    given the frequency of the antenna and the z co-ordinate
    
    Args:
        freq: Frequency of the antenna (int)
        z: z co-ordinate (int)
        comp: Component whose data must be fetched
    """
    #Folder in which the data is stored
    fname='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/'
    #Component
    fname+=comp+'_'
    #z co-ordinate
    fname+=str(z)+'m_'
    #Antenna frequency
    fname+=str(freq)+'KHz.hdf'
    
    return fname

def dataArr(filename):
    """
    This function takes an hdf5 filepath and returns the relevant data arrays
    
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
            Contains the co-ordinates for the slice (Stores all 3 co-ordinates
                                                     even though it is a slice)
    
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
    
    #Remove the unwanted component from the vertices
    #Vertices are stored in a 3 element array containing (x,y,z) but we only
    #need 2 to make flat plots, therefore we need to remove one based on the
    #plane on which the data is being plotted
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

def poynting_flux(freq,zArr):
    """
    This function plots the poynting vector flux in the xy-plane for a given
    set of z values. There is no limit to the z-values as long as all 6
    relevant components are stored in the proper folder. This function does not
    check for that condition.
    
    Since the values are in the frequency domain, to calculate the average
    Poynting vector flux, we use the following formula-
    
    avg{P}=(1/2) x Re{E x B*}
    
    Here,
    avg{P}: Average poynting vector flux
    E: Electric field in the frequency domain
    B: Magnetic field in the frequency domain
    
    The 1/2 factor is used as the average of sin^2 over one period is 1/2
    
    Args:
        freq: Frequency of the antenna (int)
        z: Array with all the z co-ordinate values needed (array)
    Returns:
        A plot of the Poynting vector flux as a function of the axial position
        in the chamber
    """
    #Initialize the array to store the flux
    PFluxArr=[]
    
    #Iterate over all the z values
    for z in zArr:

        #Create the various filenames
        filenameBx=filename(freq,z,'Bx')
        filenameBy=filename(freq,z,'By')
        filenameBz=filename(freq,z,'Bz')
        filenameEx=filename(freq,z,'Ex')
        filenameEy=filename(freq,z,'Ey')
        filenameEz=filename(freq,z,'Ez')
        
        #Get the data
        Bx,X,Y=dataArr(filenameBx)
        By,X,Y=dataArr(filenameBy)
        Bz,X,Y=dataArr(filenameBz)
        Ex,X,Y=dataArr(filenameEx)
        Ey,X,Y=dataArr(filenameEy)
        Ez,X,Y=dataArr(filenameEz)
        
        #Create the vectors
        B=[]
        E=[]
        for i in range(len(Bx)):
            B.append([Bx[i],By[i],Bz[i]])
            E.append([Ex[i],Ey[i],Ez[i]])
        #Convert to numpy arrays
        B=np.array(B)
        E=np.array(E)
        
        #Get the conjugate of B
        Bconj=np.conj(B)
        
        #Calculate the average value of the poynting vector
        avgP=[]
        for i in range(len(E)):
            currP=0.5*np.real(np.cross(E[i],Bconj[i]))
            avgP.append(currP)
        
        #Get the z values of avgP (xy plane flux is the z component of P)
        avgPZ=[]
        for P in avgP:
            avgPZ.append(P[2])
        #Convert to numpy array
        avgPZarr=np.array(avgPZ)
        
        #Interpolate the data to make integration easier
        #Create the regular mesh
        xMin=np.min(X)
        xMax=np.max(X)
        yMin=np.min(Y)
        yMax=np.max(Y)
        regX=np.linspace(xMin,xMax,int(np.round(np.sqrt(len(X)))))
        regY=np.linspace(yMin,yMax,int(np.round(np.sqrt(len(X)))))
        regX,regY=np.meshgrid(regX,regY)
        
        #Interpolation step
        interpP=interp.griddata((X,Y),avgPZarr,(regX,regY),method='linear')
        
        #Plot and save the data
        plt.contourf(regX,regY,interpP,100)
        plt.colorbar()
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.grid(True)
        plt.title('Poynting Vector Flux; f='+str(freq)+'KHz; z='+str(z)+'m')
        sliceSavepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/Pflux_XY_Plane_z='+str(z)+'m_'+str(freq)+'KHz_Ord3.png'
        plt.savefig(sliceSavepath,dpi=600,bbox_to_anchor='tight')
        plt.show()
        
        #Calculate the total flux through the XY Plane
        #Remove all nan values
        interpP=interpP[np.isfinite(interpP)]
        #Find dS
        dX=np.abs(np.abs(regX[0][0])-np.abs(regX[0][1]))
        dY=np.abs(np.abs(regY[0][0])-np.abs(regY[1][0]))
        dS=dX*dY
        #Calculate the flux
        PFlux=np.sum(interpP)
        PFlux*=dS
        
        PFluxArr.append(PFlux)
    
    return PFluxArr

#Plot the poynting flux over various z-values
PFluxArr=poynting_flux(freq,zArr)

plt.plot(zArr,PFluxArr)
plt.grid()
plt.xlabel('Z [m]')
plt.ylabel('P Flux [arb.]')
plt.title('Poynting Flux through the XY Plane as a function of Z',y=1.05)
plt.savefig(savepath,dpi=600,bbox_to_anchor='tight')
plt.show()






























