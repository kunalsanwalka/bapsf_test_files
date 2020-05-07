# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 13:56:00 2019

@author: kunalsanwalka

This program plots the By component of the data and compares the effect of damping
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.animation as animation

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

def absSquare(data):
    """
    Gets the absolute value squared of the data as the solution from Petra-M is in frequency space

    Args:
        data: numpy array with the complex, frequency space data
    Returns:
        Amplitude of the wave at that point
    """

    #Find the absolute value of the array
    absArr=data**2

    return np.abs(absArr)

def cleanData(cData,xVals,zVals,cParam=0):
    """
    This function takes the raw HDF data and cleans it up for plotting the axial amplitude

    Args:
        cData: Complex data for the By co-ordinate
        xVals,zVals: Coordinates
        cParam: Parameter that determines the data points included
    Returns:
        ampData: Array with the amplitude of the real part of By
        zData: Array with sorted z-values
    """

    #Get the absolute square of cData as we are trying the find the average over 1 wave period
    ampData=absSquare(cData)

    #Remove near field values by deleting anything with z>8m
    #Create a copy of zVals
    zValsCopy=np.copy(zVals)
    #Loop backwards
    for i in range(len(zVals)-1,-1,-1):
        if zVals[i]>8:
            zValsCopy=np.delete(zValsCopy,i)
            xVals=np.delete(xVals,i)
            ampData=np.delete(ampData,i)
    #Rewrite the new array back
    zVals=np.copy(zValsCopy)

    #Create the target grid of the interpolation
    xi=np.linspace(-5,5.01,1000)
    zi=np.linspace(-10,8.01,1000)
    xi,zi=np.meshgrid(xi,zi)

    #Interpolate the data
    interpData=griddata((xVals,zVals),ampData,(xi,zi),method='linear')

    return xi,zi,interpData

    ##Find the index of the points closest to the line at x=0
    #indexArr=[]
    #for i in range(len(xVals)):
    #    val=np.abs(xVals[i])
    #    if val<=cParam: #Closeness Parameter (arbitrary)
    #        indexArr.append(i)

    ##Get the z-position and data at the given index values
    #zData=[]
    #ampData=[]
    #for i in indexArr:
    #    zData.append(zVals[i])
    ##    ampData.append(np.sqrt(np.real(cdata[i])**2+np.imag(cdata[i])**2)) 
    #    ampData.append(cData[i])

    ##Create a dictionary with the data
    #plotDict={}
    #for i in range(len(zData)):
    #    plotDict[zData[i]]=ampData[i]
    
    ##Sort the data in ascending order
    #zData.sort()
    #ampData=[]
    #for data in zData:
    #    ampData.append(plotDict[data])

    #return ampData,zData

#User defined variables
freq=38*1000
omega=2*np.pi*freq
filename1='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/By_XZ_Plane_freq_250KHz.hdf'
filename2='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/By_XZ_Plane_freq_250KHz_col_1MHz.hdf'
filename3='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/By_XZ_Plane_freq_250KHz_col_4MHz.hdf'
savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/By_Axial_Amp_Collisional_Effects.png'
anim_savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/By_Axial_Amp.html'
plotTitle='Alfven wave amplitude as a function of Z; Antenna Location (Z)=9m'

#Get the values
cData1,xVals1,zVals1=dataArr(filename1)
cData2,xVals2,zVals2=dataArr(filename2)
cData3,xVals3,zVals3=dataArr(filename3)

#Interpolate the data
xi,zi,interpData=cleanData(cData1,xVals1,zVals1)

#Plot the amplitude of the wave
plt.contourf(xi,zi,interpData)
plt.xlim(-0.5,0.5)
plt.ylim(-10,8)
plt.colorbar()
plt.show()

#Print certain values
print(np.shape(interpData))

##Get the cleaned values
#compData1,zData1=cleanData(cData1,xVals1,zVals1,0.08)
#compData2,zData2=cleanData(cData2,xVals2,zVals2,0.08)
#compData3,zData3=cleanData(cData3,xVals3,zVals3,0.08)

##Convert the complex values into real (as the solution is in frequency space)
#ampData1=absSquare(compData1)
#ampData2=absSquare(compData2)
#ampData3=absSquare(compData3)

##Remove values for z>8.5m as they are too large and affect the rest of the plot
##Find the index of a z value greater than 8.5m
#index=len(zData1)
#for z in zData1:
#    if z>8.5:
#        index=zData1.index(z)
#        break
##Remove all values greater than 8.5 from zData and the corresponding ampData
#zData1=zData1[:index]
#zData2=zData2[:index]
#zData3=zData3[:index]
#ampData1=ampData1[:index]
#ampData2=ampData2[:index]
#ampData3=ampData3[:index]

##Plot the data
#plt.figure(figsize=(8,8))
#plt.plot(zData1,ampData1,label=r'$\nu_e$=0MHz')
#plt.plot(zData1,ampData2,label=r'$\nu_e$=1MHz')
#plt.plot(zData1,ampData3,label=r'$\nu_e$=4MHz')
#plt.yscale('symlog',linthreshy=1e-14)
#plt.grid(True)
#plt.title(plotTitle)
#plt.xlabel('Z [m]')
#plt.xlim(-10,8.5)
#plt.ylim(0,1e-10)
#plt.ylabel(r'$|B_y|^2$',labelpad=0)
#plt.legend(loc=2)
##plt.savefig(savepath,dpi=600)
#plt.show()

##Find the fourier transform of the data in kspace
##Take a log of the data to make plotting the spectrum easier
#ampData1=np.log(ampData1)
##Subtract the mean of the data from the data to isolate the oscillation
#meanVal=np.mean(ampData1)
#ampData1-=meanVal
#plt.plot(ampData1)
#plt.show()
##Find the fourier transform of the amplitude
#fftArr=np.fft.fft(ampData1)
##Find the sampling rate (average difference between points in zData)
#avgDiff=np.mean(np.diff(zData1))
#print(avgDiff)
##Find the k array
#kArr=np.fft.fftfreq(len(ampData1))*(avgDiff/2)
#plt.plot(kArr[:int(len(kArr)/2)],np.abs(fftArr)[:int(len(kArr)/2)])
#plt.show()