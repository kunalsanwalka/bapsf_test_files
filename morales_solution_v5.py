# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:41:30 2020

@author: kunalsanwalka

This program solves for the ALfven Wave structure for a wave launched via a
disk exciter. The soluion is based on a paper by Morales et. al.-

Morales, G. J., Loritsch, R. S. & Maggs, J. E. Structure of Alfvén waves at 
the skin‐depth scale. Physics of Plasmas 1, 3765–3774 (1994).
"""

import os
import numpy as np
import scipy as sc
from tqdm import tqdm
from scipy.integrate import *
import matplotlib.pyplot as plt

#Fundamental values (S.I. Units)
c=299792458             #Speed of light
eps_0=8.85418782e-12    #Vacuum Permittivity
mu_0=1.25663706212e-6   #Vacuum permeability
q_e=-1.60217662e-19     #Electron Charge
q_p=1.60217662e-19      #Proton Charge
m_e=9.10938356e-31      #Electron Mass
m_amu=1.66053906660e-27 #Atomic Mass Unit

#Plasma parameters
freq=500 #KHz #Antenna frequency
B=0.15    #T #Background magnetic field strength
n=0.5e18 #m^{-3} #Density
colE=5*1e6  #Hz #Electron collisional damping rate
colI=0 #Hz #Ion collisional damping rate

#Antenna Parameters
a=0.0025 #m #Antenna thickness
d=0.04   #m #Antenna radius
sigma0=1 #C #Antenna charge

#General savepath directory
savepath_dir='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/'
#Savepath directory from frame animations
anim_dir=savepath_dir+'morales_solution_By_XZ_Plane/'
#Create the directory
if not os.path.exists(anim_dir):
    os.makedirs(anim_dir)
#Number of frames in the animation
numFrames=60

#Savepath for variable data
data_savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/Pure Helium (for disp rel)/'
#Plot with saved data or new data
savedDataPlot=True
#Overwrite the currently saved data
saveData=True

# =============================================================================
# Derived parameters
# =============================================================================
#Frequencies
omega=2*np.pi*freq*1000 #rad/s #Angular antenna frequency
Pi_he=np.sqrt(n*q_p**2/(eps_0*4*m_amu)) #rad/s #Helium plasma frequency
Pi_e=np.sqrt(n*q_e**2/(eps_0*m_e)) #rad/s #Electron plasma frequency
Omega_he=q_p*B/(4*m_amu) #rad/s #Helium cyclotron frequency

#Velocities
va=c*(Omega_he/Pi_he)*np.sqrt(1-(omega/Omega_he)**2) #m/s #Alfven speed

#Wavenumbers
ka=omega/va #rad/m #Angular alfven wavenumber
ks=Pi_e/c #rad/m

#Current
i0=-1j*omega*sigma0*np.pi*a**2 #A #Complex Amplitude of the AC current

#Dimensionless
gammaPar=colE/omega #Parallel damping
gammaPerp=(colI/omega)+(m_e*colE/(4*m_amu*omega))*np.sqrt(1-(omega/Omega_he)**2) #Perpendicular damping

#%%
# =============================================================================
# Functions
# =============================================================================

def kPar(k):
    """
    Calculates the parallel wavenumber k_par based on the effective 
    perpendicular wavenumber k

    Args:
        k (float): Effective perpendicular wavenumber

    Returns:
        k_par (float): Parallel wavenumber
    """
    
    k_par=ka*np.sqrt(1+(k/ks)**2)
    
    return k_par

def BThetaIntegrandReal(k,r,z):
    """
    Calculates the integrand for the B_theta component of the magnetic field

    Args:
        k (float): Effective perpendicular wavenumber
        r (float): Radial position
        z (float): Axial position

    Returns:
        integrand (float): Integrand for B_theta
    """
    
    #Normalize the variables
    rho=r/a
    xi=ka*z
    K=k*a
    p=a*Pi_e/c
    
    #Corrected variables
    xiCor=xi*(1+1j*gammaPerp)
    pSqCor=(1+1j*gammaPar)/p**2
    
    sinTerm=np.sinc(K/np.pi)
    expTerm=np.e**(1j*xiCor*np.sqrt(1+pSqCor*K**2))
    besselTerm=sc.special.jv(1,K*rho)
    
    integrand=np.real(sinTerm*besselTerm*expTerm)
    
    return integrand

def BThetaIntegrandImag(k,r,z):
    """
    Calculates the integrand for the B_theta component of the magnetic field

    Args:
        k (float): Effective perpendicular wavenumber
        r (float): Radial position
        z (float): Axial position

    Returns:
        integrand (float): Integrand for B_theta
    """
    
    #Normalize the variables
    rho=r/a
    xi=ka*z
    K=k*a
    p=a*Pi_e/c
    
    #Corrected variables
    xiCor=xi*(1+1j*gammaPerp)
    pSqCor=(1+1j*gammaPar)/p**2
    
    sinTerm=np.sinc(K/np.pi)
    expTerm=np.e**(1j*xiCor*np.sqrt(1+pSqCor*K**2))
    besselTerm=sc.special.jv(1,K*rho)
    
    integrand=np.imag(sinTerm*besselTerm*expTerm)
    
    return integrand

def BTheta(r,z,omega):
    """
    Calculates the axial component of the wave magnetic field due to one disk 
    exciter in the local frame.

    Args:
        r (float): Radial position
        z (float): Axial position
        omega (float): Angular antenna frequency

    Returns:
        b_theta (complex): Theta component of the the wave magnetic field
    """    
    
    constTerm=2*i0/(c*a)
    
    #Integrate the real and imag parts seperately
    realTerm=integrator(BThetaIntegrandReal,r,z,0.001,4000,2000)
    imagTerm=integrator(BThetaIntegrandImag,r,z,0.001,4000,2000)
    
    # realTerm=quad(BThetaIntegrandReal,0.001,np.infty,args=(r,z))[0]
    # imagTerm=quad(BThetaIntegrandImag,0.001,np.infty,args=(r,z))[0]
    
    b_theta=constTerm*(realTerm+1j*imagTerm)
    
    return b_theta

def BThetaGlobal(x,y,z,d,omega):
    """
    Calculates the Bx and By component of the wave magnetic field at an
    arbitrary point due to a ring antenna (solution is in the frequency domain)

    Args:
        x,y,z (float): Position at which we are calculating the wave field
        d (float): x-displacement of the antenna
        omega (float): Antenna angular frequency

    Returns:
        Bx,By (complex): Components of the wave magnetic field
    """
    
    #Right Antenna
    #Convert to local coordinates
    theta=np.arctan2(y,x-d)
    r=np.sqrt(y**2+(x-d)**2)
    
    #Get b_theta
    b_theta=BTheta(r,z,omega)
    #Convert to Bx and By
    BxRight=-b_theta*np.sin(theta)
    ByRight=b_theta*np.cos(theta)
    
    #Left Antenna
    #Convert to local coordinates
    theta=np.arctan2(y,x+d)
    r=np.sqrt(y**2+(x+d)**2)
    
    #Get b_theta
    b_theta=-BTheta(r,z,omega)
    BxLeft=-b_theta*np.sin(theta)
    ByLeft=b_theta*np.cos(theta)
    
    Bx=BxLeft+BxRight
    By=ByLeft+ByRight
    
    return Bx,By

def integrator(funcName,r,z,lowerLim,upperLim,steps):
    """
    Calculates the integral of the BTheta integrand
    
    WARNING: Do not set lower limit to 0. You will get a divide by 0 error

    Args:
        funcName (object): Name of the function we are integrating
        r (float): Radial position
        z (float): Axial position
        lowerLim (float): Lower limit of integration
        upperLim (float): Upper limit of integration
        steps (int): Number of divisions between the upper and lower limits

    Returns:
        integral (float): Value of the integral

    """
    
    #Initialize
    integral=0
    
    #Array over which to integrate
    kArr=np.linspace(lowerLim,upperLim,steps)
    
    #Interval width
    dk=kArr[1]-kArr[0]
    
    #Starting value of the integrand
    currVal=funcName(kArr[0],r,z)
    
    #Calculate the integral
    for i in range(1,len(kArr)):
        #Current value of the integrand
        newVal=funcName(kArr[i],r,z)
        
        #Add integral over dk
        integral+=(newVal+currVal)*dk/2
        
        #Update the integrand value
        currVal=newVal
        
    return integral

def newSol(r,z,omega):
    """
    Calculates the axial component of the wave magnetic field due to one disk
    exciter in the local frame.

    Args:
        r (float): Radial position
        z (float): Axial position
        omega (float): Angular antenna frequency

    Returns:
        b_theta (complex): Theta component of the the wave magnetic field
    """
    
    #Normalized Values
    rho=r/a
    xi=ka*z
    p=a*Pi_e/c
    eta=xi/p
    
    #Scaled magnetic field
    scaledB=0
    
    #Debugging variable
    regionNum=0
    
    #Region 1
    if 0<=eta<1 and rho<=1-eta:
        den1=eta+1+np.sqrt((eta+1)**2-rho**2)
        den2=1-eta+np.sqrt((eta-1)**2-rho**2)
        
        scaledB=(rho/2)*(1/den1+1/den2)
        
        regionNum=1
    
    #Region 2
    # elif eta>=1 and rho<=1-eta:
    elif rho<=eta-1:
        den1=eta+1+np.sqrt((eta+1)**2-rho**2)
        den2=eta-1+np.sqrt((eta-1)**2-rho**2)
        
        scaledB=(rho/2)*(1/den1-1/den2)
        
        regionNum=2
        
    #Region 3
    elif rho>1-eta and rho>eta-1 and rho<=eta+1:
        den1=eta+1+np.sqrt((eta+1)**2-rho**2)
        term1=1/den1-(eta-1)/(rho**2)
        term2=np.sqrt(1-(eta-1)**2/(rho**2))
        
        scaledB=(rho/2)*term1+(1j/2)*term2
        
        regionNum=3
        
    #Region 4
    elif rho>eta+1:
        term1=1-(eta-1)**2/(rho**2)
        term2=1-(eta+1)**2/(rho**2)
        
        scaledB=1/rho+(1j/2)*(np.sqrt(term1)-np.sqrt(term2))
        
        regionNum=4
        
    return scaledB,regionNum

def newSolGlobal(x,y,z,d,omega):
    """
    Calculates the Bx and By component of the wave magnetic field at an
    arbitrary point due to a ring antenna (solution is in the frequency domain)

    Args:
        x,y,z (float): Position at which we are calculating the wave field
        d (float): x-displacement of the antenna
        omega (float): Antenna angular frequency

    Returns:
        Bx,By (complex): Components of the wave magnetic field
        regionNumL,regionNumR (int): Debugging variables
    """
    
    #Debugging values
    regionNumL=0
    regionNumR=0
    
    #Right Antenna
    #Convert to local coordinates
    theta=np.arctan2(y,x-d)
    r=np.sqrt(y**2+(x-d)**2)
    
    #Get b_theta
    # b_theta=2*i0*newSol(r,z,omega)[0]/(a*c)
    b_thetaScaled,regionNumL=newSol(r,z,omega)
    b_theta=2*i0*b_thetaScaled/(a*c)
    
    #Convert to Bx and By
    BxRight=-b_theta*np.sin(theta)
    ByRight=b_theta*np.cos(theta)
    
    #Left Antenna
    #Convert to local coordinates
    theta=np.arctan2(y,x+d)
    r=np.sqrt(y**2+(x+d)**2)
    
    #Get b_theta
    # b_theta=-2*i0*newSol(r,z,omega)[0]/(a*c)
    b_thetaScaled,regionNumR=newSol(r,z,omega)
    b_theta=-2*i0*b_thetaScaled/(a*c)
    
    #Convert to Bx and By
    BxLeft=-b_theta*np.sin(theta)
    ByLeft=b_theta*np.cos(theta)
    
    #Add the 2 antenna contributions
    Bx=BxLeft+BxRight
    By=ByLeft+ByRight
    
    return Bx,By,regionNumL,regionNumR

def createFrameXY(fieldVal,xi,yi,omega,animDir,frameNum,totFrames=numFrames):
    """
    Creates the contour plot for a specific frame

    Args:
        fieldVal (complex): 2D Array of the field value
        xi,yi (float): 2D Arrays with the positions of the data
        omega (float): Angular antenna frequency
        animDir (string): Directory to store the frames
        frameNum (int): Specific frame that is being plotted (starts at 1)
        totFrames (int, optional): Total number of frames in the animation. Defaults to numFrames.
    """
    
    #Calcualate the time in the wave period
    time=(2*np.pi/omega)*((frameNum-1)/(totFrames-1))
    
    #Exponential to go from freq to time domain
    expVal=np.e**(-1j*omega*time)
    
    #Inverse fourier transform of fieldVal
    fieldValTD=fieldVal*expVal
    
    #Take the real value (as that is what we experimentally observe)
    fieldValReal=np.real(fieldValTD)
    
    #Plot the data
    #Create the figure
    plt.figure(figsize=(10,8))
    
    #Add axes labels
    plt.title(r'$\Re(B_y)$')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    
    #Add the data
    plt.contourf(xi,yi,fieldValReal,levels=100)
    
    #Save and close
    plt.savefig(animDir+'frameNum_'+str(frameNum)+'.png',bbox_inches='tight',dpi=300)
    plt.close()

def createFrameXZ(fieldVal,xi,zi,omega,animDir,frameNum,totFrames=numFrames):
    """
    Creates the contour plot for a specific frame

    Args:
        fieldVal (complex): 2D Array of the field value
        xi,zi (float): 2D Arrays with the positions of the data
        omega (float): Angular antenna frequency
        animDir (string): Directory to store the frames
        frameNum (int): Specific frame that is being plotted (starts at 1)
        totFrames (int, optional): Total number of frames in the animation. Defaults to numFrames.
    """
    
    #Calcualate the time in the wave period
    time=(2*np.pi/omega)*((frameNum-1)/(totFrames-1))
    
    #Exponential to go from freq to time domain
    expVal=np.e**(-1j*omega*time)
    
    #Inverse fourier transform of fieldVal
    fieldValTD=fieldVal*expVal
    
    #Take the real value (as that is what we experimentally observe)
    fieldValReal=np.real(fieldValTD)
    
    #Plot the data
    #Create the figure
    plt.figure(figsize=(20,5))
    
    #Add axes labels
    plt.title(r'$\Re(B_y)$')
    plt.xlabel('Z [m]')
    plt.ylabel('X [m]')
    
    #Add the data
    plt.contourf(zi,xi,fieldValReal,levels=100)
    
    #Save and close
    plt.savefig(animDir+'frameNum_'+str(frameNum)+'.png',bbox_inches='tight',dpi=300)
    plt.close()

def createFramePerp(BxVals,ByVals,xi,yi,omega,animDir,frameNum,totFrames=numFrames):
    """
    Creates the contour plot for the perpendicular wave field component

    Args:
        BxVals,ByVals (complex): 2D Array of the field values
        xi,yi (float): 2D Arrays with the positions of the data
        omega (float): Angular antenna frequency
        animDir (string): Directory to store the frames
        frameNum (int): Specific frame that is being plotted (starts at 1)
        totFrames (int, optional): Total number of frames in the animation. Defaults to numFrames.
    """
    
    #Calcualate the time in the wave period
    time=(2*np.pi/omega)*((frameNum-1)/(totFrames-1))
    
    #Exponential to go from freq to time domain
    expVal=np.e**(-1j*omega*time)
    
    #Inverse fourier transform of fieldVal
    BxValsTD=BxVals*expVal
    ByValsTD=ByVals*expVal
    
    #Take the real value (as that is what we experimentally observe)
    BxValsReal=np.real(BxValsTD)
    ByValsReal=np.real(ByValsTD)
    
    #Find the perpendicular component
    fieldValPerp=np.sqrt(BxValsReal**2+ByValsReal**2)
    
    #Plot the data
    #Create the figure
    plt.figure(figsize=(10,8))
    
    #Add axes labels
    plt.title(r'$\Re(B_\perp)$')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    
    #Add the data
    plt.contourf(xi,yi,fieldValPerp,levels=100)
    
    #Save and close
    plt.savefig(animDir+'frameNum_'+str(frameNum)+'.png',bbox_inches='tight',dpi=300,cmap='jet')
    plt.close()

#%%
# =============================================================================
# =============================================================================
# 2D Plane Cells
# =============================================================================
# =============================================================================
    
# =============================================================================
# Create the iteration arrays
# =============================================================================

#Array length
arrLen=25

#Position arrays
xArr=np.linspace(-0.3,0.3,arrLen)
yArr=np.linspace(-0.3,0.3,arrLen)
zArr=np.linspace(0.001,20,arrLen)

#Field value arrays
bXArr=np.zeros((arrLen,arrLen),dtype=complex)
bYArr=np.zeros((arrLen,arrLen),dtype=complex)

#Debugging arrays
regionNumLArr=np.zeros((arrLen,arrLen))
regionNumRArr=np.zeros((arrLen,arrLen))

#%%
# =============================================================================
# XY Plane Solver
# =============================================================================

#XY plane z position
zDisp=2.88

#Plotting mesh
xi,yi=np.meshgrid(xArr,yArr)

#Use tqdm to get a fancy progress bar
for yInd in tqdm(range(len(yArr)),position=0):
    for xInd in range(len(xArr)):
        #Debugging
        regionNumL=0
        regionNumR=0
        
        #Get the field value
        # xVal,yVal,regionNumL,regionNumR=newSolGlobal(xArr[xInd],yArr[yInd],12.5,d,omega)
        xVal,yVal=BThetaGlobal(xArr[xInd],yArr[yInd],zDisp,d,omega)
        
        #Append data
        bXArr[yInd,xInd]=xVal
        bYArr[yInd,xInd]=yVal
        regionNumLArr[yInd,xInd]=regionNumL
        regionNumRArr[yInd,xInd]=regionNumR

if saveData==True:
    #Save the data
    np.save(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_By_XY_Plane.npy',bYArr)
    np.save(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_Bx_XY_Plane.npy',bXArr)
    np.save(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_yi_XY_Plane.npy',yi)
    np.save(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_xi_XY_Plane.npy',xi)

#%%
# =============================================================================
# XZ Plane Solver
# =============================================================================

#Plotting mesh
zi,xi=np.meshgrid(zArr,xArr)

#Use tqdm to get a fancy progress bar
for xInd in tqdm(range(len(xArr)),position=0):
    for zInd in range(len(zArr)):
        #Debugging
        regionNumL=0
        regionNumR=0
        
        #Get the field value
        # xVal,yVal,regionNumL,regionNumR=newSolGlobal(xArr[xInd],0,zArr[zInd],d,omega)
        xVal,yVal=BThetaGlobal(xArr[xInd],0,zArr[zInd],d,omega)
        
        #Append data
        bXArr[xInd,zInd]=xVal
        bYArr[xInd,zInd]=yVal
        regionNumLArr[xInd,zInd]=regionNumL
        regionNumRArr[xInd,zInd]=regionNumR

if saveData==True:
    #Save the data
    np.save(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_By_XZ_Plane.npy',bYArr)
    np.save(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_Bx_XZ_Plane.npy',bXArr)
    np.save(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_zi_XZ_Plane.npy',zi)
    np.save(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_xi_XZ_Plane.npy',xi)

#%%
# =============================================================================
# Animation XY
# =============================================================================

#Animation
for i in range(numFrames):
    print('Creating frame '+str(i+1)+' of '+str(numFrames))
    createFramePerp(bXArr,bYArr,xi,yi,omega,anim_dir,i+1)

#%%
# =============================================================================
# Animation XZ
# =============================================================================

#Animation
for i in range(numFrames):
    print('Creating frame '+str(i+1)+' of '+str(numFrames))
    createFrameXZ(bYArr,xi,zi,omega,anim_dir,i+1)
    
#%%
# =============================================================================
# XY Plane Plotter
# =============================================================================

if savedDataPlot==True:
    bYArr=np.load(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_By_XY_Plane.npy')
    bXArr=np.load(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_Bx_XY_Plane.npy')
    yi=np.load(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_yi_XY_Plane.npy')
    xi=np.load(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_xi_XY_Plane.npy')
    
    #Normalize the axes
    xiNorm=xi*Pi_e/c
    yiNorm=yi*Pi_e/c

    #Bx plot
    plt.figure(figsize=(8,8))
    plt.title(r'$\Re(B_x)$')
    plt.xlabel(r'$x\omega_{pe}/c$')
    plt.ylabel(r'$y\omega_{pe}/c$')
    plt.xlim(-60,60)
    plt.ylim(-60,60)
    plt.contourf(xiNorm,yiNorm,np.real(bXArr),levels=100,cmap='jet')
    plt.savefig(savepath_dir+'Bx_XY_Plane_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()
    
    #By plot
    plt.figure(figsize=(8,8))
    plt.title(r'$\Re(B_y)$')
    plt.xlabel(r'$x\omega_{pe}/c$')
    plt.ylabel(r'$y\omega_{pe}/c$')
    plt.xlim(-60,60)
    plt.ylim(-60,60)
    plt.contourf(xiNorm,yiNorm,np.real(bYArr),levels=100,cmap='jet')
    plt.savefig(savepath_dir+'By_XY_Plane_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()
    
else:
    #Normalize the axes
    xiNorm=xi*Pi_e/c
    yiNorm=yi*Pi_e/c
    
    #Bx plot
    plt.figure(figsize=(8,8))
    plt.title(r'$\Re(B_x)$')
    plt.xlabel(r'$x\omega_{pe}/c$')
    plt.ylabel(r'$y\omega_{pe}/c$')
    plt.xlim(-60,60)
    plt.ylim(-60,60)
    plt.contourf(xiNorm,yiNorm,np.real(bXArr),levels=100,cmap='jet')
    plt.savefig(savepath_dir+'Bx_XY_Plane_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()
    
    #By plot
    plt.figure(figsize=(8,8))
    plt.title(r'$\Re(B_y)$')
    plt.xlabel(r'$x\omega_{pe}/c$')
    plt.ylabel(r'$y\omega_{pe}/c$')
    plt.xlim(-60,60)
    plt.ylim(-60,60)
    plt.contourf(xiNorm,yiNorm,np.real(bYArr),levels=100,cmap='jet')
    plt.savefig(savepath_dir+'By_XY_Plane_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()
    
#%%
# =============================================================================
# XZ Plane Plotter
# =============================================================================

if savedDataPlot==True:
    #Load the data
    bYArr=np.load(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_By_XZ_Plane.npy')
    bXArr=np.load(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_Bx_XZ_Plane.npy')
    zi=np.load(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_zi_XZ_Plane.npy')
    xi=np.load(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_xi_XZ_Plane.npy')

    #Normalize the axes
    xiNorm=xi*Pi_e/c
    ziNorm=zi*Pi_e/c

    #Bx plot
    plt.figure(figsize=(16,8))
    plt.title(r'$\Re(B_x)$')
    plt.xlabel(r'$z\omega_{pe}/c$')
    plt.ylabel(r'$x\omega_{pe}/c$')
    plt.xlim(0,2500)
    plt.ylim(-60,60)
    plt.contourf(ziNorm,xiNorm,np.real(bXArr),levels=100,cmap='jet')
    plt.savefig(savepath_dir+'Bx_XZ_Plane_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()
    
    #By plot
    plt.figure(figsize=(16,8))
    plt.title(r'$\Re(B_y)$')
    plt.xlabel(r'$z\omega_{pe}/c$')
    plt.ylabel(r'$x\omega_{pe}/c$')
    plt.xlim(0,2500)
    plt.ylim(-60,60)
    plt.contourf(ziNorm,xiNorm,np.real(bYArr),levels=100,cmap='jet')
    plt.savefig(savepath_dir+'By_XZ_Plane_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()

else:
    #Normalize the axes
    xiNorm=xi*Pi_e/c
    ziNorm=zi*Pi_e/c
    
    #Bx plot
    plt.figure(figsize=(16,8))
    plt.title(r'$\Re(B_x)$')
    plt.xlabel(r'$z\omega_{pe}/c$')
    plt.ylabel(r'$x\omega_{pe}/c$')
    plt.xlim(0,2500)
    plt.ylim(-60,60)
    plt.contourf(ziNorm,xiNorm,np.real(bXArr),levels=100,cmap='jet')
    plt.savefig(savepath_dir+'Bx_XZ_Plane_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()
    
    #By plot
    plt.figure(figsize=(16,8))
    plt.title(r'$\Re(B_y)$')
    plt.xlabel(r'$z\omega_{pe}/c$')
    plt.ylabel(r'$x\omega_{pe}/c$')
    plt.xlim(0,2500)
    plt.ylim(-60,60)
    plt.contourf(ziNorm,xiNorm,np.real(bYArr),levels=100,cmap='jet')
    plt.savefig(savepath_dir+'By_XZ_Plane_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()

#%%
# =============================================================================
# =============================================================================
# 1D Lineplot Cells
# =============================================================================
# =============================================================================

# =============================================================================
# Z Axis Solver
# =============================================================================

#Array length
arrLen=500

#Position array
zArr=np.linspace(0.001,20,arrLen)

#Field Value arrays
bXArr=np.zeros(arrLen,dtype='complex')
bYArr=np.zeros(arrLen,dtype='complex')

#Use tqdm to get a fancy progress bar
for zInd in tqdm(range(len(zArr)),position=0):
    
    #Get the field values
    xVal,yVal=BThetaGlobal(0,0,zArr[zInd],d,omega)
    
    #Append data
    bXArr[zInd]=xVal
    bYArr[zInd]=yVal
    
if saveData==True:
    #Save the data
    np.save(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_By_Z_lineplot.npy',bYArr)
    np.save(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_Bx_Z_lineplot.npy',bXArr)
    np.save(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_zArr_Z_lineplot.npy',zArr)

#%%
# =============================================================================
# X Axis Solver
# =============================================================================

#X lineplot z position
zDisp=1 #m

#Array length
arrLen=100

#Position array
xArr=np.linspace(-0.3,0.3,arrLen)

#Field Value arrays
bXArr=np.zeros(arrLen,dtype='complex')
bYArr=np.zeros(arrLen,dtype='complex')

#Use tqdm to get a fancy progress bar
for xInd in tqdm(range(len(xArr)),position=0):
    
    #Get the field values
    xVal,yVal=BThetaGlobal(xArr[xInd],0,zDisp,d,omega)
    
    #Append data
    bXArr[xInd]=xVal
    bYArr[xInd]=yVal
    
if saveData==True:
    #Save the data
    np.save(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_By_X_lineplot_z_'+str(int(zDisp*100))+'cm.npy',bYArr)
    np.save(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_Bx_X_lineplot_z_'+str(int(zDisp*100))+'cm.npy',bXArr)
    np.save(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_xArr_X_lineplot_z_'+str(int(zDisp*100))+'cm.npy',xArr)

#%%
# =============================================================================
# Z Axis Plotter
# =============================================================================

#Z Axis Limits
zLim=(0,2500)

if savedDataPlot==True:
    #Load the data
    bYArr=np.load(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_By_Z_lineplot.npy')
    bXArr=np.load(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_Bx_Z_lineplot.npy')
    zArr=np.load(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_zArr_Z_lineplot.npy')

    #Normalize the axes
    zNorm=zArr*Pi_e/c
    
    #Bx Plot
    plt.figure(figsize=(10,8))
    plt.xlabel(r'$z\omega_{pe}/c$')
    plt.ylabel(r'$\Re(B_x)$')
    plt.xlim(zLim[0],zLim[1])
    plt.plot(zNorm,np.real(bXArr))
    plt.grid(True)
    plt.savefig(savepath_dir+'Bx_Z_lineplot_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()
    
    #By Plot
    plt.figure(figsize=(10,8))
    plt.xlabel(r'$z\omega_{pe}/c$')
    plt.ylabel(r'$\Re(B_y)$')
    plt.xlim(zLim[0],zLim[1])
    plt.plot(zNorm,np.real(bYArr))
    plt.grid(True)
    plt.savefig(savepath_dir+'By_Z_lineplot_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()

else:
    #Normalize the axes
    zNorm=zArr*Pi_e/c
    
    #Bx Plot
    plt.figure(figsize=(10,8))
    plt.xlabel(r'$z\omega_{pe}/c$')
    plt.ylabel(r'$\Re(B_x)$')
    plt.xlim(zLim[0],zLim[1])
    plt.plot(zNorm,np.real(bXArr))
    plt.grid(True)
    plt.savefig(savepath_dir+'Bx_Z_lineplot_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()
    
    #By Plot
    plt.figure(figsize=(10,8))
    plt.xlabel(r'$z\omega_{pe}/c$')
    plt.ylabel(r'$\Re(B_y)$')
    plt.xlim(zLim[0],zLim[1])
    plt.plot(zNorm,np.real(bYArr))
    plt.grid(True)
    plt.savefig(savepath_dir+'By_Z_lineplot_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()

#%%
# =============================================================================
# X Axis Plotter
# =============================================================================

if savedDataPlot==True:
    #Load the data
    bYArr=np.load(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_By_X_lineplot.npy')
    BxArr=np.load(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_Bx_X_lineplot.npy')
    xArr=np.load(data_savepath+'freq_'+str(freq)+'KHz_col_'+str(int(colE/1e3))+'KHz_xArr_X_lineplot.npy')

    #Normalize the axes
    xNorm=xArr*Pi_e/c
    
    #Bx Plot
    plt.figure(figsize=(10,8))
    plt.xlabel(r'$x\omega_{pe}/c$')
    plt.ylabel(r'$\Re(B_x)$')
    plt.xlim(-60,60)
    plt.plot(xNorm,np.real(bXArr))
    plt.grid(True)
    plt.savefig(savepath_dir+'Bx_X_lineplot_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()
    
    #By Plot
    plt.figure(figsize=(10,8))
    plt.xlabel(r'$x\omega_{pe}/c$')
    plt.ylabel(r'$\Re(B_y)$')
    plt.xlim(-60,60)
    plt.plot(xNorm,np.real(bYArr))
    plt.grid(True)
    plt.savefig(savepath_dir+'By_X_lineplot_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()

else:
    #Normalize the axes
    xNorm=xArr*Pi_e/c
    
    #Bx Plot
    plt.figure(figsize=(10,8))
    plt.xlabel(r'$x\omega_{pe}/c$')
    plt.ylabel(r'$\Re(B_x)$')
    plt.xlim(-60,60)
    plt.plot(xNorm,np.real(bXArr))
    plt.grid(True)
    plt.savefig(savepath_dir+'Bx_X_lineplot_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()
    
    #By Plot
    plt.figure(figsize=(10,8))
    plt.xlabel(r'$x\omega_{pe}/c$')
    plt.ylabel(r'$\Re(B_y)$')
    plt.xlim(-60,60)
    plt.plot(xNorm,np.real(bYArr))
    plt.grid(True)
    plt.savefig(savepath_dir+'By_X_lineplot_freq_'+str(freq)+'KHz_moralesSolution.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()
    
#%%
# =============================================================================
# Frequency Sweeps
# =============================================================================

# =============================================================================
# Z Axis Solver
# =============================================================================

#Frequency array
freqArr=np.arange(1,575,5)

#Go over every frequency
for freq in tqdm(freqArr,position=0):
	
	#Redefine a bunch of the variables
	#Frequencies
	omega=2*np.pi*freq*1000 #rad/s #Angular antenna frequency
	
	#Velocities
	va=c*(Omega_he/Pi_he)*np.sqrt(1-(omega/Omega_he)**2) #m/s #Alfven speed
	
	#Wavenumbers
	ka=omega/va #rad/m #Angular alfven wavenumber
	ks=Pi_e/c #rad/m
	
	#Current
	i0=-1j*omega*sigma0*np.pi*a**2 #A #Complex Amplitude of the AC current
	
	#Dimensionless
	gammaPar=colE/omega #Parallel damping
	gammaPerp=(colI/omega)+(m_e*colE/(4*m_amu*omega))*np.sqrt(1-(omega/Omega_he)**2) #Perpendicular damping
	
	#Solve for the z-axis
	#Array length
	arrLen=600
	
	#Position array
	zArr=np.linspace(0.001,80,arrLen)
	
	#Field Value arrays
	bXArr=np.zeros(arrLen,dtype='complex')
	bYArr=np.zeros(arrLen,dtype='complex')
	
	#Use tqdm to get a fancy progress bar
	for zInd in range(len(zArr)):
	    
	    #Get the field values
	    xVal,yVal=BThetaGlobal(0,0,zArr[zInd],d,omega)
	    
	    #Append data
	    bXArr[zInd]=xVal
	    bYArr[zInd]=yVal
	    
	if saveData==True:
	    #Save the data
	    np.save(data_savepath+'freq_'+str(int(freq))+'KHz_col_'+str(int(colE/1e3))+'KHz_By_Z_lineplot.npy',bYArr)
# 	    np.save(data_savepath+'freq_'+str(int(freq))+'KHz_col_'+str(int(colE/1e3))+'KHz_Bx_Z_lineplot.npy',bXArr)
	    np.save(data_savepath+'freq_'+str(int(freq))+'KHz_col_'+str(int(colE/1e3))+'KHz_zArr_Z_lineplot.npy',zArr)


#%%
# =============================================================================
# Integrand plot
# =============================================================================

#Angular wavenumber array
kArr=np.linspace(0.0001,4000,2000)

#r and z values (arb.)
r=0.5
z=0.1

#Integrand arrays
integrandReal=np.zeros(len(kArr))
integrandImag=np.zeros(len(kArr))

#Calculate the arrays
for i in range(len(kArr)):
    integrandReal[i]=BThetaIntegrandReal(kArr[i],r,z)
    integrandImag[i]=BThetaIntegrandImag(kArr[i],r,z)
    
#Plot the result
plt.figure(figsize=(8,8))
plt.plot(kArr,integrandReal,label='Real')
plt.plot(kArr,integrandImag,label='Imag')
plt.xlabel('k [rad/m]')
plt.ylabel('Integrand Amp')
plt.xlim(0,500)
plt.legend()
plt.grid(True)
plt.show()
plt.close()