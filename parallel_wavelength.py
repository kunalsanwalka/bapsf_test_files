# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:26:30 2020

@author: kunalsanwalka

File to calculate the parallel wavelength of the Alfven Wave
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.interpolate import griddata

###############################################################################
#############################User defined variables############################
###############################################################################
freqArr=[37,80,108,206,275,343,378,446,480,500,530,560] #Array with all the frequencies

#Data Directory
data_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/New Density/'

#Savepath Location
savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/angular_wavenumber_multiIon.png'
###############################################################################

#Plasma Parameters
magB=0.15               #T      #Magnetic Field Stength
ne=1e18                 #m^{-3} #Plasma Density
nu_e=1e6                #Hz     #Plasma Collisionality
Omega_Ne=115.17         #Hz     #Neon Cyclotron Frequency
Omega_He=575.90         #Hz     #Helium Cyclotron Frequency
heRatio=0.5             #arb.   #Helium Fraction
neRatio=0.5             #arb.   #Neon Fraction
kPerp=2*np.pi/0.08      #m^{-1} #Angular perpendicular wavenumber

#Fundamental values (S.I.)
c=299792458             #Speed of light
eps_0=8.85418782e-12    #Vacuum Permittivity
q_e=-1.60217662e-19     #Electron Charge
q_p=1.60217662e-19      #Proton Charge
m_e=9.10938356e-31      #Electron Mass
m_amu=1.66053906660e-27 #Atomic Mass Unit

#%%
# =============================================================================
# Functions
# =============================================================================

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

def ByAxialAmp(dataArr,zi):
    """
    This function takes the interpolated data and returns the value of By along
    the r=0 line.
    
    Args:
        dataArr: Interpolated data of By
        zi: 2D array with the z values
    Returns:
        ByAmp: Amplitude of By along r=0
        zVals: Array containing the z-axis values
    """
    
    #Get the length of the z-array
    zlen=len(zi)
    
    #Transpose to get values along the z-axis
    zTrans=np.transpose(zi)
    ByTrans=np.transpose(dataArr)
    
    #Get the z-axis values
    zVals=zTrans[int(zlen/2)]
    
    #Get the By values
    ByAmp=ByTrans[int(zlen/2)]
    
    return ByAmp,zVals

def fourierTransform(ByVals,zVals):
    """
    This function finds the fourier transform of the data.
    
    Note- The kArr returned by this function is the linear wavenumber. To
          convert to the angular wavenumber, you need to multiply the result
          by 2*pi elsewhere in the code
    
    Args:
        ByVals: By values along r=0
        zVals: z-axis values
    Returns:
        ByF: Fourier transformed By data
        kArr: Array with the corresponding k values
    """
    
    #Remove all nan values from ByVals
    #Locate all the nan values
    nanIndArr=np.argwhere(np.isnan(ByVals))
    #Remove the values at those indices
    for ind in nanIndArr[::-1]:
        #print('data len'+str(len(data)))
        ByVals=np.delete(ByVals,ind[0])
        zVals=np.delete(zVals,ind[0])
    
    #Length of the simulation space
    simLen=np.max(zVals)-np.min(zVals)
    
    #Sampling rate (number of samples per meter)
    sRate=len(zVals)/simLen
    
    #Take the fourier transform
    ByF=np.abs(fftpack.fft(ByVals))
    #Set k=0 amplitude to 0 to get rid of any DC effects
    ByF[0]=0
    #Create the wavenumber array
    kArr=fftpack.fftfreq(len(ByVals))*sRate
    
    #Only return the first half of the array since ByVals is real
    ByF=ByF[:int(len(ByF)/2)]
    kArr=kArr[:int(len(kArr)/2)]
    
    return ByF,kArr
    
def peakWavenumber(ByF,kArr):
    """
    This function takes the fourier transformed data and finds the dominant
    wavenumber.
    
    Args:
        ByF: Fourier transformed data
        kArr: Array with the wavenumbers
    Returns:
        peakK: Most prominent wavenumber
    """
    
    #Find the location of the maximum value of ByF
    maxVal=np.argmax(ByF)
    
    #Find the corresponding wavenumber
    peakK=kArr[maxVal]
    
    return peakK

def LHPdispRel(w):
    """
    Calculates the left handed polarized wave dispersion relation

    Args:
        w (INT): Angular frequency of the antenna
    Returns:
        kPar (INT): Angular wavenumber
    """
    #Calcuate the relevant frequencies
    Pi_he=np.sqrt((ne*heRatio)*q_p**2/(eps_0*4*m_amu)) #Helium plasma frequency
    Pi_ne=np.sqrt((ne*neRatio)*q_p**2/(eps_0*20*m_amu)) #Neon plasma frequency
    Pi_e=np.sqrt(ne*q_e**2/(eps_0*m_e)) #Electron plasma frequency
    Omega_he=q_p*magB/(4*m_amu) #Helium cyclotron frequency
    Omega_ne=q_p*magB/(20*m_amu) #Neon cyclotron frequency
    Omega_e=q_e*magB/(m_e) #Electron cyclotron frequency
   
    #Calculate L
    L=1-((Pi_e**2/w**2)*(w/(w-Omega_e)))-((Pi_he**2/w**2)*(w/(w-Omega_he)))-((Pi_ne**2/w**2)*(w/(w-Omega_ne)))
    
    kPar=np.sqrt((L*w**2)/(c**2))
    
    return kPar

def RHPdispRel(w):
    """
    Calculates the right handed polarized wave dispersion relation

    Args:
        w (INT): Angular frequency of the antenna
    Returns:
        kPar (INT): Angular wavenumber
    """
    #Calcuate the relevant frequencies
    Pi_he=np.sqrt((ne*heRatio)*q_p**2/(eps_0*4*m_amu)) #Helium plasma frequency
    Pi_ne=np.sqrt((ne*neRatio)*q_p**2/(eps_0*20*m_amu)) #Neon plasma frequency
    Pi_e=np.sqrt(ne*q_e**2/(eps_0*m_e)) #Electron plasma frequency
    Omega_he=q_p*magB/(4*m_amu) #Helium cyclotron frequency
    Omega_ne=q_p*magB/(20*m_amu) #Neon cyclotron frequency
    Omega_e=q_e*magB/(m_e) #Electron cyclotron frequency
   
    #Calculate L
    R=1-((Pi_e**2/w**2)*(w/(w+Omega_e)))-((Pi_he**2/w**2)*(w/(w+Omega_he)))-((Pi_ne**2/w**2)*(w/(w+Omega_ne)))
    
    kPar=np.sqrt((R*w**2)/(c**2))
    
    return kPar

def finiteKPerpdispRel(w):
    """
    Calculates the finite k_perp dispersion relation

    Args:
        w (INT): Angular frequency of the antenna
    Returns:
        kPar (INT): Angular wavenumber
    """
    #Various plasma frequencies
    Pi_he=np.sqrt((ne*heRatio)*q_p**2/(eps_0*4*m_amu)) #Helium plasma frequency
    Pi_ne=np.sqrt((ne*neRatio)*q_p**2/(eps_0*20*m_amu)) #Neon plasma frequency
    Pi_e=np.sqrt(ne*q_e**2/(eps_0*m_e)) #Electron plasma frequency
    Omega_he=q_p*magB/(4*m_amu) #Helium cyclotron frequency
    Omega_ne=q_p*magB/(20*m_amu) #Neon cyclotron frequency
    Omega_e=q_e*magB/(m_e) #Electron cyclotron frequency
    
    #R,L and P
    R=1-((Pi_e**2/w**2)*(w/(w+Omega_e)))-((Pi_he**2/w**2)*(w/(w+Omega_he)))-((Pi_ne**2/w**2)*(w/(w+Omega_ne))) #Right-hand polarized wave
    L=1-((Pi_e**2/w**2)*(w/(w-Omega_e)))-((Pi_he**2/w**2)*(w/(w-Omega_he)))-((Pi_ne**2/w**2)*(w/(w-Omega_ne))) #Left-hand polarized wave
    P=1-(Pi_e**2/(w*(w+1j*nu_e)))-(Pi_he**2/w**2)-(Pi_ne**2/w**2) #Unmagnetized plasma

    #S and D
    S=(R+L)/2
    D=(R-L)/2
    
    #u=w**2/c**2
    u=(w/c)**2
    
    #g_perp=k_perp**2
    gPerp=kPerp**2
    
    #Cubic equation coefficients
    bTerm=(gPerp*S/P)+(2*gPerp)-(L*u)-(R*u)
    cTerm=(2*gPerp*gPerp*S/P)-(gPerp*R*L*u/P)-(gPerp*S*u)+(gPerp*gPerp)-(gPerp*L*u)-(gPerp*R*u)+(R*L*u*u)
    dTerm=(gPerp*gPerp*gPerp*S/P)-(gPerp*gPerp*R*L*u/P)-(gPerp*gPerp*S*u)+(gPerp*R*L*u*u)
    
    #Depressed cubic equation coefficients
    pTerm=(3*cTerm-bTerm*bTerm)/3
    qTerm=(2*bTerm*bTerm*bTerm-9*bTerm*cTerm+27*dTerm)/27
    
    #kPar
    kPar=0
    if 4*pTerm*pTerm*pTerm+27*qTerm*qTerm>0:
        #Single real root
        term1=(-qTerm/2+np.sqrt((qTerm*qTerm/4)+(pTerm*pTerm*pTerm/27)))**(1/3)
        term2=(-qTerm/2-np.sqrt((qTerm*qTerm/4)+(pTerm*pTerm*pTerm/27)))**(1/3)
        realRoot=term1+term2
        
        #Convert back to original cubic
        gPar=realRoot-bTerm/3
        
        #Calcualte kPar
        kPar=np.sqrt(gPar)
        
    else:
        #arccos term
        arccosTerm=np.arccos((3*qTerm/(2*pTerm))*np.sqrt(-3/pTerm))
        #cos term
        k=0
        cosTerm=np.cos((1/3)*arccosTerm-2*np.pi*k/3)
        
        #Real root
        realRoot=2*np.sqrt(-pTerm/3)*cosTerm
        
        #Convert back to original cubic
        gPar=realRoot-bTerm/3
        
        #Calcualte kPar
        kPar=np.sqrt(gPar)
    
    return kPar

#%%
# =============================================================================
# Simulation Dispersion Relationm
# =============================================================================

#Array to store the wavenumbers
kParArr=[]

#Go over each frequency to get the parallel angular wavenumber
for freq in freqArr:
    
    #Print message in the Console
    print('Working on frequency- '+str(freq)+'KHz')
    
    #Define the data location
    datapath=data_dir+'By_XZ_Plane_freq_'+str(freq)+'KHz.hdf'
    
    #Get the data
    By,X,Z=dataArr(datapath)
    
    #Set time t=0
    By=np.real(By)
    
    #Interpolate the data
    ByInterp,xi,zi=interpData(By,X,Z)
    
    #Get the value along r=0
    ByAmp,zVals=ByAxialAmp(ByInterp,zi)
    
    #Remove all values where z<0.2m
    for z in zVals[::-1]:
        if z<0.2:
            ByAmp=np.delete(ByAmp,np.where(zVals==z))
            zVals=np.delete(zVals,np.where(zVals==z))
    
    #Perform the fourier transform
    ByF,kArr=fourierTransform(ByAmp,zVals)
    
    #Find the peak wavenumber
    wavenum=peakWavenumber(ByF,kArr)
    
    #Convert to angular wavenumber
    kPar=2*np.pi*wavenum
    
    #Add to the array
    kParArr.append(kPar)
    
#Normalize the frequency
normFreqArr=np.array(freqArr)/Omega_Ne

#%%
# =============================================================================
# Analytical Dispersion Relation
# =============================================================================
 
#Array to store the wavenumbers
lhpArr=[]
rhpArr=[]
finiteKPerpArr=[]

#Array with the angular frequencies
omegaArr=2*np.pi*np.linspace(100,(Omega_He-1)*1000,1000)

#Calculate the angular wavenumber
for omega in omegaArr:
    lhpArr.append(LHPdispRel(omega))
    rhpArr.append(RHPdispRel(omega))
    finiteKPerpArr.append(finiteKPerpdispRel(omega))
    
#Normalize the frequency
normFreqArrAnal=omegaArr/(2*np.pi*Omega_Ne*1000) #Becuase Omega_Ne is linear

#%%
# =============================================================================
# Plot the data
# =============================================================================

#Define the axes labels
plt.title(r'B=0.15T; n=$5 \cdot 10^{17} m^{-3}$; 50/50 He/Ne')
plt.xlabel(r'Normalized Angular Frequency [$\omega/\Omega_{Ne}$]')
plt.ylabel(r'Angular Wavenumber [$k_{||}$]')

#Plot the data
plt.scatter(normFreqArr,kParArr,label='Simulation Data',color='red')
plt.plot(normFreqArrAnal,finiteKPerpArr,label=r'$k_{\perp}$='+str(np.round(kPerp,2)))
plt.plot(normFreqArrAnal,lhpArr,label='LHP',linestyle='dashed')
plt.plot(normFreqArrAnal,rhpArr,label='RHP',linestyle='dashed')

#Plot the normalized frequencies
plt.plot([1,1],[0,max(kParArr)+0.5],label=r'$\Omega_{Ne}$',color='k',linestyle='-')
plt.plot([Omega_He/Omega_Ne,Omega_He/Omega_Ne],[0,max(kParArr)+0.5],label=r'$\Omega_{He}$',color='k',linestyle=':')

#Miscellaneous
plt.grid(True)
plt.xlim(0,Omega_He/Omega_Ne+0.1)
plt.ylim(0,max(kParArr)+0.25)

#Add the legend
plt.legend(bbox_to_anchor=(1.4,1.03),loc='upper right')

#Show and close the plot
plt.savefig(savepath,dpi=600,bbox_inches='tight')
plt.show()
plt.close()

# #Find the wavelength of each wave
# By,X,Z=dataArr('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/By_XZ_Plane_freq_480KHz_col_5000KHz.hdf')
# ByInterp,xi,zi=interpData(By,X,Z)
# ByAmp,zVals=ByAxialAmp(ByInterp,zi)

# #Remove all values where z<0.2m
# for z in zVals[::-1]:
#     if z<0.2:
#         ByAmp=np.delete(ByAmp,np.where(zVals==z))
#         zVals=np.delete(zVals,np.where(zVals==z))

# ByF,kArr=fourierTransform(ByAmp,zVals)

# plt.plot(kArr,ByF)
# plt.show()

# print(peakWavenumber(ByF,kArr))