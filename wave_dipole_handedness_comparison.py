# -*- coding: utf-8 -*-
"""
@author: kunalsanwalka

This program calculates the ratio of the handedness of the alfven wave and plots it as a function of the frequency
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':13})
import h5py
from scipy.interpolate import griddata

###############################################################################
###########################  User defined variables  ##########################
###############################################################################

#Frequencies at which the simulation was run
#Note: This is the only variable defined with a linear frequency.
#      Every other frequency variable uses angular frequency.
freqArr=[80,108,206,275,343,378,446,480,500,530,560]

#Plasma Parameters
#Magnetic field
magB=0.15 #Tesla
#Density of the plasma
ne=0.5e18 #m^{-3}
#Collisionality of the plasma
nu_e=1e6 #Hz

#Data Directory
dataDir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/New Density/'

#Savepath location
savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/handedness_ratio_perpWavelength.png'

###############################################################################

#Fundamental values
c=299792458             #Speed of light
eps_0=8.85418782e-12    #Vacuum Permittivity
q_e=-1.60217662e-19     #Electron Charge
q_p=1.60217662e-19      #Proton Charge
m_e=9.10938356e-31      #Electron Mass
m_amu=1.66053906660e-27 #Atomic Mass Unit

#Calculate the cyclotron frequency
#Helium cyclotron frequency
cycFreqHe=q_p*magB/(4*m_amu) #rad/s
#Neon cyclotron frequency
cycFreqNe=q_p*magB/(20*m_amu) #rad/s

#Calculate the plasma frequency
Pi_e=np.sqrt((ne*q_e**2)/(eps_0*m_e)) #Electron plasma frequency

#Calculate the plasma skin depth
plasmaSD=c/Pi_e

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
    Interpolates the data into a regualar grid to allow for cleaner plots and simpler data manipulation

    Args:
        ampData: The raw data
        xVals,yVals: The co-ordinates of the data
    Returns:
        interpData: The interpolated data
        xi,yi: Meshgrid to allow for the plotting of the data
    """

    #Create the target grid of the interpolation
    xi=np.linspace(-0.5,0.5,201)
    yi=np.linspace(-0.5,0.5,201)
    xi,yi=np.meshgrid(xi,yi)

    #Interpolate the data
    interpData=griddata((xVals,yVals),ampData,(xi,yi),method='linear')

    return interpData,xi,yi

def multiIonPolRatio(w,lambda_perp,magB):
    """
    This function calculates the analytical polarization ratio based on the
    solution for a multi-ion plasma
    
    Args:
        w: Angular frequency of the antenna
        lambda_perp: Perpendicular wavelength of the wave
        magB: Magnitude of the magnetic field
    Returns:
        polR: Polarization Ratio
    """ 
    #Ratio between the gases
    heRatio=0.5
    neRatio=0.5    
    
    #Calcuate the relevant frequencies
    Pi_he=np.sqrt((ne*heRatio)*q_p**2/(eps_0*4*m_amu)) #Helium plasma frequency
    Pi_ne=np.sqrt((ne*neRatio)*q_p**2/(eps_0*20*m_amu)) #Neon plasma frequency
    Pi_e=np.sqrt(ne*q_e**2/(eps_0*m_e)) #Electron plasma frequency
    Omega_he=q_p*magB/(4*m_amu) #Helium cyclotron frequency
    Omega_ne=q_p*magB/(20*m_amu) #Neon cyclotron frequency
    Omega_e=q_e*magB/(m_e) #Electron cyclotron frequency
   
    #Calculate R,L and P
    R=1-((Pi_e**2/w**2)*(w/(w+Omega_e)))-((Pi_he**2/w**2)*(w/(w+Omega_he)))-((Pi_ne**2/w**2)*(w/(w+Omega_ne))) #Right-hand polarized wave
    L=1-((Pi_e**2/w**2)*(w/(w-Omega_e)))-((Pi_he**2/w**2)*(w/(w-Omega_he)))-((Pi_ne**2/w**2)*(w/(w-Omega_ne))) #Left-hand polarized wave
    P=1-(Pi_e**2/(w*(w+1j*nu_e)))-(Pi_he**2/w**2)-(Pi_ne**2/w**2) #Unmagnetized plasma

    #Calculate S and D
    S=(R+L)/2
    D=(R-L)/2

    #Perpendicular refractive index
    #Perpendicular wavenumber
    k_perp=2*np.pi/lambda_perp
    n_perp=c*k_perp/w
    
    #Find B_l and B_r
    Bl=(n_perp**2)*(1-L/P)-D
    Br=(n_perp**2)*(1-R/P)+D
    
    #Find the polarization ratio
    polR=np.abs(Bl/Br)**2
    
    return polR

def squareDiff(arr1,arr2):
    """
    This function calculates the square difference between 2 different
    arrays
    
    Args:
        arr1,arr2: 2 arrays for which we need the least squares difference  
    Returns:
        chiSq: The sum of the square difference
    """
    #Convert both arrays to numpy
    arr1=np.array(arr1)
    arr2=np.array(arr2)
    
    #Find the difference between the 2 arrays
    arrDiff=arr1-arr2
    
    #Square the difference
    diffSq=arrDiff**2
    
    #Sum the difference
    chiSq=np.sum(diffSq)
    
    return chiSq

#Sort the frequencies in ascending order
freqArr.sort()
#Convert to numpy array
freqArr=np.array(freqArr)

#Array with the normalized frequencies (w.r.t. Neon)
normFreqArr=2*np.pi*freqArr*1000/cycFreqNe

#Array with the names of files with the Bx and By data
filenameX=[]
filenameY=[]
for freq in freqArr:
    BxString='Bx_XY_Plane_freq_'+str(freq)+'KHz.hdf'
    ByString='By_XY_Plane_freq_'+str(freq)+'KHz.hdf'
    filenameX.append(dataDir+BxString)
    filenameY.append(dataDir+ByString)

#Get the ratio of the handedness
ratioArr=[]
for i in range(len(filenameX)):

    #Get the data
    Bx,X,Y=dataArr(filenameX[i])
    By,X,Y=dataArr(filenameY[i])

    #Left Handed Wave
    BLeft=(1/np.sqrt(2))*(Bx+1j*By)
    BLeftAbs=np.abs(BLeft)**2
    #Right Handed Wave
    BRight=(1/np.sqrt(2))*(Bx-1j*By)
    BRightAbs=np.abs(BRight)**2

    #Interpolate the data
    BLeftInterpTemp,xi,yi=interpData(BLeftAbs,X,Y)
    BRightInterpTemp,xi,yi=interpData(BRightAbs,X,Y)

    #Replace all nan values with 0
    BLeftInterp=np.nan_to_num(BLeftInterpTemp)
    BRightInterp=np.nan_to_num(BRightInterpTemp)
    
    # #Find the ratio at the center
    # ratio=BLeftInterp[100][100]/BRightInterp[100][100]
    
    #Integrate over one axis
    tempIntegralLeft=np.trapz(BLeftInterp,axis=0)
    tempIntegralRight=np.trapz(BRightInterp,axis=0)
    #Integrate over the other axis
    integralLeft=np.trapz(tempIntegralLeft)
    integralRight=np.trapz(tempIntegralRight)

    #Get the ratio of the handedness
    ratio=integralLeft/integralRight
    
    #Append the ratio to the array
    ratioArr.append(ratio)

#Get the value of k_perp via a least squares optimization
#Array with the test lambda_perp values
optLambdaPerpArr=np.linspace(2,120,1000)*plasmaSD
#Best lambda_perp
bestLambdaPerp=optLambdaPerpArr[0]
#Best chiSq
bestChiSq=100000000000
#Go over each lambda_perp
for lambdaPerp in optLambdaPerpArr:
    #Array to store the analytical polarization ratio
    currAnalRatio=[]
    #Go over each frequency in the simulations
    for freq in freqArr:
        #Convert to angular frequency
        w=2*np.pi*freq*1000
        #Add ratio to the array
        currAnalRatio.append(multiIonPolRatio(w,lambdaPerp,magB))
    #Find the least squares difference
    currChiSq=squareDiff(currAnalRatio,ratioArr)
    
    #Check and replace
    if currChiSq<bestChiSq:
        bestChiSq=currChiSq
        #Divide by plasmaSD as later we multiple by plasmaSD again
        bestLambdaPerp=lambdaPerp/plasmaSD

#Get the analytical polarization curves for different k_perp values
#Frequency array to iterate over
analFreqArr=np.linspace(1,cycFreqHe,5000)
#Perpendicular wavelengths (in m)
lambdaPerpArr=np.array([2,bestLambdaPerp,120])*plasmaSD
#Store the polarization ratios
analRatio=[]
for i in range(len(lambdaPerpArr)):
    analRatio.append([])
    for w in analFreqArr:
        analRatio[i].append(multiIonPolRatio(w,lambdaPerpArr[i],magB))
#Convert to numpy
analFreqArr=np.array(analFreqArr)
analRatio=np.array(analRatio)
lambdaPerpArr=np.array(lambdaPerpArr)

#Normalize the arrays for plotting
#Frequency
analFreqArr/=cycFreqNe
#Perpendicular wavelength
lambdaPerpArr/=plasmaSD

# =============================================================================
# Plot the data
# =============================================================================

#Make the plot
p1=plt.figure()

#Plot the analytical solutions
for i in range(len(lambdaPerpArr)):
    p1=plt.plot(analFreqArr,analRatio[i],label=r'$\lambda_{\perp}/\delta_e=$'+str(np.round(lambdaPerpArr[i],2)))

#Plot the simulation data
plt.scatter(normFreqArr,ratioArr,label='Simulation Data')

#Plot the cyclotron frequencies
plt.plot([1,1],[min(ratioArr)-0.01,max(ratioArr)+0.01],label=r'$\Omega_{Neon}$',color='k',linestyle='-',linewidth=2)
plt.plot(np.full(2,cycFreqHe/cycFreqNe),[min(ratioArr)-0.01,max(ratioArr)+0.01],label=r'$\Omega_{Helium}$',color='k',linestyle=':')

#Add the labels and title
plt.title(r'B=0.15T; n=$5 \cdot 10^{17} m^{-3}$; 50/50 He/Ne')
plt.xlabel(r'Normalized Angular Frequency [$\omega/\Omega_{Neon}$]')
plt.ylabel(r'Polarization Ratio [$|B_L/B_R|^2$]',rotation=90)

#Miscellaneous
plt.ylim(min(ratioArr)-0.01,max(ratioArr)+0.01)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.53,1.03),loc='upper right')
plt.savefig(savepath,dpi=600,bbox_inches='tight')
plt.show()
plt.close()