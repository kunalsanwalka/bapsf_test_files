# -*- coding: utf-8 -*-
"""
@author: kunalsanwalka

This program plots the By component of the data and other relations
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

def cleanData(cData,xVals,zVals):
    """
    This function takes the raw HDF data and cleans it up for plotting the axial amplitude

    Args:
        cData: Complex data for the By co-ordinate
        xVals,zVals: Coordinates
    Returns:
        interpData: Array with the interpolated values of |By**2|
        xi,zi: Meshgrid to allow for plotting of the data
    """

    #Get the real part of cData (as that is what we experimentally observe)
    ampData=np.real(cData)

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
    xi=np.linspace(-0.5,0.5,1000)
    zi=np.linspace(-12,8.0,1000)
    xi,zi=np.meshgrid(xi,zi)

    #Interpolate the data
    interpData=griddata((xVals,zVals),ampData,(xi,zi),method='linear')

    return xi,zi,interpData

def runFuncs(filename):
    """
    This function takes the filename and runs the various functions on it to give the final data/results

    Args:
        filename: Name of the file (str)
    Returns:
        xi,zi: The interpolated XZ Plane
        interpData: The interpolated values
    """
    #Get the values
    cData,xVals,zVals=dataArr(filename)

    #Interpolate the data
    xi,zi,interpData=cleanData(cData,xVals,zVals)

    #Transpose to the get the amplitude along the z-line
    interpData=np.transpose(interpData)

    return xi,zi,interpData

def peakWavenumber(data,zVals,dataLen,PMLPos):
    """
    This function takes in the data array and returns the angular wavenumber of the alfven wave

    Args:
        data: The 2D array with the values of By at t=0 (np.array)
        zVals: z-axis values corresponding to data (np.array)
        dataLen: The length of the simulation space over which data is defined (float)
        PMLPos: Position at which the PML region starts (float)
    Returns:
        waveNum: Angular wavenumber of the alfven wave (float)
    """

    #Remove the parts of data that are in the PML region
    #Shorten zVals and data (assumes PML is in the negative region)
    for z in zVals[::-1]: #Traverse the list backwards
        if z<PMLPos:
            data=np.delete(data,np.where(zVals==z))
            zVals=np.delete(zVals,np.where(zVals==z))

    #print(len(data))
    #print(len(zVals))

    #Find out where all the nan values in data are
    nanIndArr=np.argwhere(np.isnan(data))
    #Remove the values at those indices
    for ind in nanIndArr[::-1]:
        #print('data len'+str(len(data)))
        data=np.delete(data,ind[0])
        zVals=np.delete(zVals,ind[0])

    #Grid spacing/Sampling rate
    samplingRate=np.sqrt(2*np.pi)/(dataLen/len(data)) #2*pi as k is the angular wavenumber

    #Take the fourier transform
    kAmp=np.abs(fftpack.fft(data))
    #Set the DC value as 0
    kAmp[0]=0
    #Create the wavenumber array
    waveNumArr=fftpack.fftfreq(len(data))*samplingRate

    #plt.figure(figsize=(6,6))
    #plt.title(r'Fourier transform of $B_y$ in k-space')
    #plt.xlabel(r'Parallel Wavenumber $k_{||}$ [rad/m]')
    #plt.ylabel('Amplitude [arb u.]')
    #plt.yscale('log')
    #plt.grid(True)
    #plt.plot(waveNumArr[:len(kAmp)//2],kAmp[:len(kAmp)//2])
    #plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/By_fourier_transform.png',bbox_to_anchor='tight',dpi=700)
    #plt.show()

    #Find the peak of the transform (wavelength of the alfven wave)
    peakind=np.argmax(kAmp[:int(len(waveNumArr)/2)])
    #Find the corresponding wavenumber (k)
    waveNum=waveNumArr[:int(len(waveNumArr)/2)][peakind]

    #k=2*pi/lambda
    return waveNum*np.sqrt(2*np.pi)

def MHDDispRel(omega,magB,numDens):
    """
    This function calculates the angular wavenumber of an alfven wave based on the MHD model

    Args:
        omega: Angular frequency
        magB: Magnitude of the background magnetic field
        numDens: Number density of the particles
    Returns:
        kPar: Parallel angular wavenumber
    """

    #Phase velocity
    va=magB/np.sqrt(const.mu_0*numDens*const.physical_constants['alpha particle mass'][0])

    #Angular Wavenumber
    kPar=omega/va

    return kPar

#Simulation Variables
magB=0.1 #T
numDens=1e18 #m^{-3}

#Directories
data_directory='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/'
savepath_directory='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/'

#Filenames
#Mesh=6cm
filenameArr=['By_XZ_Plane_freq_50KHz_mesh60.hdf','By_XZ_Plane_freq_100KHz_mesh60.hdf','By_XZ_Plane_freq_150KHz_mesh60.hdf','By_XZ_Plane_freq_200KHz_mesh60.hdf','By_XZ_Plane_freq_250KHz_mesh60.hdf','By_XZ_Plane_freq_275KHz_mesh60.hdf','By_XZ_Plane_freq_300KHz_mesh60.hdf','By_XZ_Plane_freq_350KHz_mesh60.hdf']
#Mesh=7.5cm
#filenameArr=['By_XZ_Plane_freq_50KHz_mesh75.hdf','By_XZ_Plane_freq_100KHz_mesh75.hdf','By_XZ_Plane_freq_150KHz_mesh75.hdf','By_XZ_Plane_freq_200KHz_mesh75.hdf','By_XZ_Plane_freq_250KHz_mesh75.hdf','By_XZ_Plane_freq_275KHz_mesh75.hdf','By_XZ_Plane_freq_300KHz_mesh75.hdf','By_XZ_Plane_freq_350KHz_mesh75.hdf']
#Mesh=10cm
#filenameArr=['By_XZ_Plane_freq_50KHz.hdf','By_XZ_Plane_freq_100KHz.hdf','By_XZ_Plane_freq_150KHz.hdf','By_XZ_Plane_freq_200KHz.hdf','By_XZ_Plane_freq_250KHz.hdf','By_XZ_Plane_freq_275KHz.hdf','By_XZ_Plane_freq_300KHz.hdf','By_XZ_Plane_freq_350KHz.hdf']
savepath='By_Axial_Amp.png'
#savepath='By_Axial_Amp_Collisional_Effects.png'
fftsavepath='By_fourier_transform.png'
#fftsavepath='By_fourier_transform.png'
kparsavepath='k_par_dispersion_relation.png'
kperpsavepath='k_perp_dispersion_relation.png'

#Plot titles/labels
#ampPlotTitle='Alfven wave amplitude as a function of Z; Frequency='+str(freq/1000)+'KHz; Antenna Location (Z)=9m'
ampPlotTitle='Alfven wave amplitude as a function of Z; Antenna Location (Z)=9m'
labelArr=['50KHz','100KHz','150KHz','200KHz','250KHz','275KHz','300KHz','350KHz']
#labelArr=[r'$\nu_e$=0MHz',r'$\nu_e$=1MHz',r'$\nu_e$=4MHz']

#Create array with the filenames
filenameArrCopy=filenameArr
filenameArr=[]
for filename in filenameArrCopy:
    filenameArr.append(data_directory+filename)

#Add the directory label to the savepaths
savepath=savepath_directory+savepath
fftsavepath=savepath_directory+fftsavepath
kparsavepath=savepath_directory+kparsavepath
kperpsavepath=savepath_directory+kperpsavepath

#Get the interpolated values
#Initialize variables
interpDataArr=[]
xi=0
zi=0
for filename in filenameArr:
    print(filename)
    xi,zi,interpData=runFuncs(filename)
    interpDataArr.append(interpData)

##Contour plot of the 1st data file
#plt.figure(figsize=(16,4))
#plt.contourf(xi,zi,interpDataArr[2])
#plt.show()

#Get the x and z values
zAxisVals=np.transpose(zi)[0]
xAxisVals=xi[0]

#Find the index which is closest to x=0
#Get the absolute value of the x values
xArrAbs=np.abs(xi[0])
#Convert to list
xList=xArrAbs.tolist()
#Find the index of the minimum value
minInd=xList.index(min(xList))

#Plot the amplitude along z for x=0
p1=plt.figure(figsize=(8,8))
for i in range(len(interpDataArr)):
    p1=plt.plot(zAxisVals,interpDataArr[i][minInd],label=labelArr[i])
p1=plt.title(ampPlotTitle,y=1.015)
p1=plt.xlabel('Z [m]')
p1=plt.ylabel(r'$\Re(B_y)$')
p1=plt.xlim(-12,8)
p1=plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
p1=plt.xticks(np.arange(min(zAxisVals), max(zAxisVals)+1, 1.0))
p1=plt.grid()
p1=plt.legend(loc='best')
#p1=plt.savefig(savepath,dpi=600,bbox_to_anchor='tight')
p1=plt.show()

#Find the k_|| dispersion relation

#Initalize Arrays
freqArr=np.array([50,100,150,200,250,275,300,350])*2*np.pi*1000 #KHz #2*pi as we want the angular frequency
waveNumArr=[] #Stores k_|| obtained from the simulation
MHDArr=[] #Stores k_|| obtained from MHD Theory
twoFluidArr=[] #Stores k_|| obtained from 2-fluid Theory

#Find the MHD Dispersion Relation
for omega in freqArr:
    MHDArr.append(MHDDispRel(omega,magB,numDens))

#Find the simulation dispersion relation
#Go over the various data arrays
for i in range(len(interpDataArr)):
    dataGrid=interpDataArr[i]
    #Find k_par for every x-value
    currKPar=[]
    for j in range(len(dataGrid)):
        #Ignore the density falloff regions
        if j>400 and j<600:
            currKPar.append(peakWavenumber(dataGrid[j],zAxisVals,20,-10))
    #Find the average value and append it to the array
    waveNumArr.append(sum(currKPar)/len(currKPar))

#for i in range(len(interpDataArr)):
#    dataGrid=interpDataArr[i]
#    waveNumArr.append(peakWavenumber(dataGrid[minInd],zAxisVals,20,-10))

#Plot the dispersion relation
p2=plt.plot(waveNumArr,freqArr,label='Simulated Dispersion Relation')
p2=plt.plot(MHDArr,freqArr,label='MHD Dispersion Relation')
p2=plt.plot(waveNumArr,np.full(len(waveNumArr),383.9*1000*2*np.pi),label='Cyclotron Frequency')
p2=plt.title(r'Alfven Wave Dispersion Relation for $k_{||}$')
p2=plt.xlabel(r'Angular Wavenumber$(k_{||})$ [rad/m]')
p2=plt.ylabel(r'Angular Frequency$(\omega)$ [rad/s]')
p2=plt.ylim(0,2.5e6)
p2=plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
p2=plt.grid(True)
p2=plt.legend(loc='best')
#p2=plt.savefig(kparsavepath,dpi=600,bbox_to_anchor='tight')
p2=plt.show()

#Save the data in a text file
with open('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/k_par_mesh_60.txt','w') as f:
    for ele in waveNumArr:
        f.write('%s\n'%str(ele))

#Find the k_perp dispersion relation

#Find the plasma skin depth
del_e=5.31e3/np.sqrt(numDens) #m
#Find the predicted alfven wave phase velocity
vAlfvenMHD=magB/np.sqrt(const.mu_0*numDens*const.physical_constants['alpha particle mass'][0])

#Find the alfven phase velocity array
vAlfvenSim=freqArr/np.array(waveNumArr) #y-axis

#Create the array to store k_perp
kPerpArr=[] #x-axis

#Find the k_perp values
for i in range(len(interpDataArr)):
    #Transpose so we can pull z-slices easily
    dataGrid=np.transpose(interpDataArr[i])

    #Find k_perp for every z-value
    currKPerp=[]
    for j in range(len(dataGrid)):
        #Ignore the PML region
        if j>100 and j<950:
            currKPerp.append(peakWavenumber(dataGrid[j],xAxisVals,1,-10))
    #Find the average value and append it to the array
    kPerpArr.append(sum(currKPerp)/len(currKPerp))
#Convert kPerpArr to a numpy array
kPerpArr=np.array(kPerpArr)

#Plot each plane of data for debugging purposes
for i in range(len(interpDataArr)):
    dataGrid=interpDataArr[i]
    xlen=len(dataGrid)
    ylen=len(dataGrid[0])
    x=np.linspace(-12,8,xlen)
    y=np.linspace(-0.5,0.5,ylen)
    X,Y=np.meshgrid(x,y)
    plt.figure(figsize=(10,2))
    plt.contourf(X,Y,dataGrid,levels=100)
    plt.xlabel('Z [m]')
    plt.ylabel('X [m]')
    plt.title('Contour plot of $\Re(B_y)$ on the XZ-Plane')
    plt.colorbar()
    plt.gcf().subplots_adjust(bottom=0.25)
    #plt.yticks(np.arange(0,1000,50))
    #plt.grid(True)
    plt.savefig(savepath_directory+str(np.round(freqArr[i]/(2*np.pi*1000),0))+'KHz.png',dpi=600)
    plt.show()

#Plot the dispersion relation
p3=plt.plot(kPerpArr*del_e,vAlfvenSim,label='Dispersion Relation')
p3=plt.plot(kPerpArr*del_e,np.full(len(kPerpArr),vAlfvenMHD),label='MHD Phase Velocity')
p3=plt.title(r'Alfven Wave Dispersion Relation for $k_{\perp}$')
p3=plt.xlabel(r'Normalized Angular Wavenumber$(k_{\perp}\delta_e)$ [arb.]')
p3=plt.ylabel('Alfven Wave Phase Velocity$(v_A)$ [m/s]')
p3=plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
p3=plt.grid(True)
p3=plt.legend(loc='best')
#p3=plt.savefig(kperpsavepath,dpi=600,bbox_to_anchor='tight')
p3=plt.show()

#Save the data in a text file
with open('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/k_perp_mesh_60.txt','w') as f:
    for ele in kPerpArr:
        f.write('%s\n'%str(ele))