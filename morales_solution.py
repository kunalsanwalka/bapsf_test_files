# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:41:30 2020

@author: kunalsanwalka
"""

import numpy as np
import scipy as sc
from scipy.integrate import quad
import matplotlib.pyplot as plt

#Fundamental values (S.I. Units)
c=299792458             #Speed of light
eps_0=8.85418782e-12    #Vacuum Permittivity
q_e=-1.60217662e-19     #Electron Charge
q_p=1.60217662e-19      #Proton Charge
m_e=9.10938356e-31      #Electron Mass
m_amu=1.66053906660e-27 #Atomic Mass Unit

#Plasma parameters
freq=192 #KHz #Antenna frequency
B=0.1 #T #Background magnetic field strength
n=1.4e18 #m^{-3} #Density
a=0.025 #m #Antenna radius
sigma0=1 #C #Antenna charge

#Derived parameters
#Frequencies
omega=2*np.pi*freq*1000 #rad/s #Angular antenna frequency
Pi_he=np.sqrt(n*q_p**2/(eps_0*4*m_amu)) #rad/s #Helium plasma frequency
Pi_e=np.sqrt(n*q_e**2/(eps_0*m_e)) #rad/s #Electron plasma frequency
Omega_he=q_p*B/(4*m_amu) #rad/s #Helium cyclotron frequency
#Velocities
va=c*(Omega_he/Pi_he)*np.sqrt(1-(omega/Omega_he)**2) #m/s #Conventional alfven speed
#Wavenumbers
ka=omega/va #rad/m #Angular alfven wavenumber
ks=Pi_e/c #rad/m

#%%
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

def BThetaIntegrand(k,r,z):
    """
    Calculates the integrand for the B_theta component of the magnetic field

    Args:
        k (float): Effective perpendicular wavenumber
        r (float): Radial position
        z (float): Axial position

    Returns:
        integrand (float): Integrand for B_theta
    """
    
    sinTerm=np.sin(k*a)/k
    expTerm=np.e**(1j*z*kPar(k))
    besselTerm=sc.special.jv(1,k*r)
    
    integrand=sinTerm*besselTerm*expTerm
    
    return integrand

def BTheta(r,z,omega):
    """
    Calculates the axial component of the wave magnetic field

    Args:
        r (float): Radial position
        z (float): Axial position
        omega (float): Angular antenna frequency

    Returns:
        b_theta (float): DESCRIPTION.
    """    
    
    i0=-1j*omega*sigma0*np.pi*a**2
    constTerm=2*i0/(c*a)
    integrandTerm=quad(BThetaIntegrand,0,np.inf,args=(r,z))
    
    b_theta=constTerm*integrandTerm
    
    return b_theta

#%%

# =============================================================================
# Calculate the value
# =============================================================================
        
#Position arrays
xArr=np.linspace(0,10*a,100)
yArr=np.linspace(0,10*a,100)

