# -*- coding: utf-8 -*-
"""
@author: kunalsanwalka

This program calculates the fourier transform of the wave over one wave period
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.interpolate import griddata
from scipy import fftpack

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

def cleanData(cData,xVals,zVals,phase):
    """
    This function takes the raw HDF data and cleans it up for plotting the axial amplitude

    Args:
        cData: Complex data for the By co-ordinate
        xVals,zVals: Coordinates
        phase: Phase of the wave for which we are calculating the interpolation
    Returns:
        interpData: Array with the interpolated values of |By**2|
        xi,zi: Meshgrid to allow for plotting of the data
    """

    #Get the real part of cData at a given phase (as that is what we experimentally observe)
    ampData=np.real(cData*phase)

    #Remove near field values by deleting anything with z>8m
    #Create a copy of zVals
    zValsCopy=np.copy(zVals)

    #Loop backwards
    for i in range(len(zVals)-1,-1,-1):
        if zVals[i]>2.1 or zVals[i]<-3.1:
            zValsCopy=np.delete(zValsCopy,i)
            xVals=np.delete(xVals,i)
            ampData=np.delete(ampData,i)
    #Rewrite the new array back
    zVals=np.copy(zValsCopy)

    #Create the target grid of the interpolation
    xi=np.linspace(-0.5,0.5,1000)
    zi=np.linspace(-3,2.0,1000)
    xi,zi=np.meshgrid(xi,zi)

    #Interpolate the data
    interpData=griddata((xVals,zVals),ampData,(xi,zi),method='linear')

    return xi,zi,interpData

def frequencySpace(zVals,data):
    """
    This function converts the wave plot from real space to frequency space
    Note: the k=0 value is set to 0

    Args:
        zVals: z-axis values
        data: Wave amplitude
    Returns:
        waveNumArr[:len(kAmp)//2]: Frequency axis
        kAmp[:len(kAmp)//2]: Amplitude of the frequency
    """

    #Find out where all the nan values in data are
    nanIndArr=np.argwhere(np.isnan(data))
    #Remove the values at those indices
    for ind in nanIndArr[::-1]:
        #print('data len'+str(len(data)))
        data=np.delete(data,ind[0])
        zVals=np.delete(zVals,ind[0])

    #Grid spacing/Sampling rate
    samplingRate=(2*np.pi)/(5.25/len(data)) #2*pi as k is the angular wavenumber

    #Take the fourier transform
    kAmp=np.abs(fftpack.fft(data))
    #Set the DC value as 0
    kAmp[0]=0
    #Create the wavenumber array
    waveNumArr=fftpack.fftfreq(len(data))*samplingRate

    return waveNumArr[:len(kAmp)//2],kAmp[:len(kAmp)//2]

def runFuncs(filename,omega):
    """
    This function takes the filename and runs the various functions on it to give the final data/results

    Args:
        filename: Name of the file (str)
        omega: Angular frequency of the wave
    Returns:
        kParArr: Array with the k_|| values
        fftAmpArr: Amplitude of the fourier transform
    """
    #Get the values from the hdf5 file
    cData,xVals,zVals=dataArr(filename)

    #Time array
    tArr=np.linspace(0,2*np.pi/omega,30)
    #Phase Array
    expArr=np.exp(-1j*omega*tArr)

    #Initialize fourier transform arrays
    kParArr=np.zeros(500)
    fftAmpArr=np.zeros(500)

    #Go through each phase value
    for phase in expArr:
        #************************************************************#
        #Helps track code progress
        print(str(np.where(expArr==phase)[0][0]+1)+'/'+str(len(expArr))+':- '+str(phase))
        #************************************************************#
        
        #Interpolate the data
        xi,zi,interpData=cleanData(cData,xVals,zVals,phase)

        #Transpose to the get the amplitude along the z-line
        interpData=np.transpose(interpData)

        #Find the index which is closest to x=0
        #Get the absolute value of the x values
        xArrAbs=np.abs(xi[0])
        #Convert to list
        xList=xArrAbs.tolist()
        #Find the index of the minimum value
        minInd=xList.index(min(xList))

        #Find the wave amplitude along x=0
        waveAmpArr=interpData[minInd]
        #Get the z-axis values
        zAxisVals=np.transpose(zi)[0]

        #Calculate the fourier transform
        kParTemp,fftAmpTemp=frequencySpace(zAxisVals,waveAmpArr)
        #Convert to numpy arrays
        kParTemp=np.array(kParTemp)
        fftAmpTemp=np.array(fftAmpTemp)
        #Add to the final fourier transform
        kParArr+=kParTemp
        fftAmpArr+=fftAmpTemp

        #Plot the wave
        plt.plot(zAxisVals,waveAmpArr)
        plt.title(str(np.where(expArr==phase)[0][0]))
        plt.grid(True)
        #plt.savefig('expArr['+str(np.where(expArr==phase)[0][0])+']')
        plt.close()

    #Normalize with the arrays
    kParArr/=len(tArr)
    fftAmpArr/=len(tArr)

    return kParArr,fftAmpArr

filename='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/By_XZ_Plane_freq_275KHz_mesh4cm.hdf'
freq=275*1000
omega=2*np.pi*freq

#Get the fourier transform
kParArr,fftAmpArr=runFuncs(filename,omega)

#Find the peak k_|| value
peakInd=np.where(fftAmpArr==max(fftAmpArr))[0][0]
peakKPar=kParArr[peakInd]

plt.plot(kParArr,fftAmpArr)
plt.grid(True)
plt.title(r'Time Averaged Fourier Transform; $k_{||}$='+str(np.round(peakKPar,2)),y=1.03)
plt.xlabel(r'Angular Wavenumber $k_{||}$')
plt.xlim(0,100)
plt.ylabel('Magnitude')
plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/time_avg_fourier_transform.png',dpi=600)
plt.show()
plt.close()