# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:45:45 2020

@author: kunalsanwalka

This program solves for the Alfven Wave structure for a wave launched via an
electric dipole antenna. The solution is based on the thesis by Jeffrey
Robertson.
"""

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
B=0.15    #T #Background magnetic field strength
n=1e18 #m^{-3} #Density
colE=5*1e6  #Hz #Electron collisional damping rate
colI=0 #Hz #Ion collisional damping rate

# =============================================================================
# Derived parameters
# =============================================================================
#Frequencies
Pi_he=np.sqrt(n*q_p**2/(eps_0*4*m_amu)) #rad/s #Helium plasma frequency
Pi_e=np.sqrt(n*q_e**2/(eps_0*m_e)) #rad/s #Electron plasma frequency
Omega_he=q_p*B/(4*m_amu) #rad/s #Helium cyclotron frequency

#Lengths
plasmaSD=c/Pi_e #Plasma Skin Depth

#Antenna parameters
freq=0.75*Omega_he #KHz #Antenna frequency
omega=2*np.pi*freq*1000 #rad/s #Angular antenna frequency
antI=1 #A #Dipole current
l=20*plasmaSD #m #Dipole length

#%%
# =============================================================================
# Functions
# =============================================================================

def dipoleVacField(r,z):
    """
    This function calculates the vacuum magnetic field due to an electric
    dipole antenna.
    
    Args:
        r,z (float): Position at which we want the vaccum field value
    
    Returns:
        Bvec (np.array): Magnetic field vector in cylindrical coordinates
    """
    
    #Initialize the magnetic field components
    Br=0 #Radial component
    Bt=0 #Azimuthal component
    Bz=0 #z component
    
    #Calculate the azimuthal magnetic field component
    constTerm=mu_0*antI/(4*np.pi*r)
    geomTerm1=(z+l/2)/np.sqrt(r**2+(z+l/2)**2)
    geomTerm2=(z-l/2)/np.sqrt(r**2+(z-l/2)**2)
    
    Bt=constTerm*(geomTerm1-geomTerm2)
    
    Bvec=np.array([Br,Bt,Bz])
    
    return Bvec

def hankelTransformIntegrand(funcName,kPerp,r,z):
    """
    This function calculates the integrand for the Hankel transform
    
    NOTE: Used for integrating 'by hand'
    
    Args:
        funcName (object): Name of the function to be Hankel transformed
        kPerp (float): Perpendicular angular wavenumber
        r,z (float): Position at which to calculate the Hankel transform
    """
    
    return funcName(r,z)*sc.special.jv(1,kPerp*r)*r

def hankelTransformIntegrandBr(r,kPerp,funcName,z):
    """
    This function calculates the Br integrand for the Hankel transform
    
    NOTE: Used for integrating using scipy quad
    
    Args:
        funcName (object): Name of the function to be Hankel transformed
        kPerp (float): Perpendicular angular wavenumber
        r,z (float): Position at which to calculate the Hankel transform
    """
    
    return funcName(r,z)[0]*sc.special.jv(1,kPerp*r)*r

def hankelTransformIntegrandBt(r,kPerp,funcName,z):
    """
    This function calculates the Bt integrand for the Hankel transform
    
    NOTE: Used for integrating using scipy quad
    
    Args:
        funcName (object): Name of the function to be Hankel transformed
        kPerp (float): Perpendicular angular wavenumber
        r,z (float): Position at which to calculate the Hankel transform
    """
    
    return funcName(r,z)[1]*sc.special.jv(1,kPerp*r)*r

def hankelTransformIntegrandBz(r,kPerp,funcName,z):
    """
    This function calculates the Bz integrand for the Hankel transform
    
    NOTE: Used for integrating using scipy quad
    
    Args:
        funcName (object): Name of the function to be Hankel transformed
        kPerp (float): Perpendicular angular wavenumber
        r,z (float): Position at which to calculate the Hankel transform
    """
    
    return funcName(r,z)[2]*sc.special.jv(1,kPerp*r)*r

def hankelTransform(funcName,kPerp,z,lowerLim,upperLim,steps):
    """
    This function calculates the Hankel transform of the vaccum magnetic field
    of the antenna
    
    WARNING: Do not set lower limit to 0. You will get a divide by 0 error

    Args:
        funcName (object): Name of the function to be Hankel transformed
        kPerp (float): Perpendicular angular wavenumber
        z (float): Z position at which to calculate the Hankel transform
        lowerLim (float): Lower limit of integration
        upperLim (float): Upper limit of integration
        steps (int): Number of divisions between the upper and lower limits

    Returns:
        BrHank,BtHank,BzHank (np.array): Hankel transformed components of the 
                                         antenna magnetic field
    """
    
    #Initialize
    BrHank=0
    BtHank=0
    BzHank=0
    
    #Array over which to integrate
    rArr=np.linspace(lowerLim,upperLim,steps)
    
    #Interval width
    dr=rArr[1]-rArr[0]
    
    #Starting value of the integrand
    currVal=hankelTransformIntegrand(funcName,kPerp,rArr[0],z)
    
    #Calculate the integral
    for i in range(1,len(rArr)):
        #Current value of the integrand
        newVal=hankelTransformIntegrand(funcName,kPerp,rArr[i],z)
        
        #Add integral over dk
        BrHank+=(newVal[0]+currVal[0])*dr/2
        BtHank+=(newVal[1]+currVal[1])*dr/2
        BzHank+=(newVal[2]+currVal[2])*dr/2
        
        #Update the integrand value
        currVal=newVal
    
    #Integrate using scipy
    BrHank=quad(hankelTransformIntegrandBr,0,np.infty,args=(kPerp,funcName,z))[0]
    # BtHank=quad(hankelTransformIntegrandBt,0,np.infty,args=(kPerp,funcName,z))[0]
    # BzHank=quad(hankelTransformIntegrandBz,0,np.infty,args=(kPerp,funcName,z))[0]
    
    return np.array([BrHank,BtHank,BzHank])

#%%
# =============================================================================
# Debugging
# =============================================================================
    
#Normalized kPerp array
kPerpNorm=np.logspace(-3,2,500)

#Store the hankel transforms
BrHankArr=[]
BtHankArr=[]

#Calculate the Hankel transformed values
for i in tqdm(range(len(kPerpNorm)),position=0):
    
    hankTrans=hankelTransform(dipoleVacField,kPerpNorm[i]/plasmaSD,0,0.001,100,5000)
    
    BrHankArr.append(hankTrans[0])
    BtHankArr.append(hankTrans[1])
    
#Normalize for plotting
BrHankArr=np.array(BrHankArr)
BtHankArr=np.array(BtHankArr)

BrHankArr=2*np.pi*kPerpNorm*np.abs(BrHankArr)/(mu_0*antI*plasmaSD)
BtHankArr=2*np.pi*kPerpNorm*np.abs(BtHankArr)/(mu_0*antI*plasmaSD)

#%%
# =============================================================================
# Plotting
# =============================================================================

plt.plot(kPerpNorm,BrHankArr,label='r')
plt.plot(kPerpNorm,BtHankArr,label=r'$\theta$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-3,10**2)
plt.ylim(10**-7,2)
plt.grid()
plt.xlabel(r'$k_{\perp}\delta_e$')
plt.ylabel('Norm val')
plt.legend()
plt.show()
plt.close() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    