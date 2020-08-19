# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:45:45 2020

@author: kunalsanwalka

This program solves for the Alfven Wave structure for a wave launched via an
electric dipole antenna. The solution is based on the thesis by Jeffrey
Robertson.
"""

import cmath
import numpy as np
import scipy as sc
from tqdm import tqdm
from scipy.integrate import *
import matplotlib.pyplot as plt

#Fundamental values (S.I. Units)
c=299792458			 #Speed of light
eps_0=8.85418782e-12	#Vacuum Permittivity
mu_0=1.25663706212e-6   #Vacuum permeability
q_e=-1.60217662e-19	 #Electron Charge
q_p=1.60217662e-19	  #Proton Charge
m_e=9.10938356e-31	  #Electron Mass
m_amu=1.66053906660e-27 #Atomic Mass Unit

#Plasma parameters
B=0.15	#T #Background magnetic field strength
n=1e18 #m^{-3} #Density
nu_e=0 #Hz #General damping term
colE=5*1e6  #Hz #Electron collisional damping rate
colI=0 #Hz #Ion collisional damping rate

# =============================================================================
# Derived parameters
# =============================================================================
#Frequencies
Pi_he=np.sqrt(n*q_p**2/(eps_0*4*m_amu)) #rad/s #Helium plasma frequency
Pi_e=np.sqrt(n*q_e**2/(eps_0*m_e)) #rad/s #Electron plasma frequency
Omega_he=q_p*B/(4*m_amu) #rad/s #Helium cyclotron frequency
Omega_e=q_e*B/m_e

#Lengths
plasmaSD=c/Pi_e #Plasma Skin Depth

#Antenna parameters
omega=0.75*Omega_he #rad/s #Angular antenna frequency
antI=1 #A #Dipole current
l=20*plasmaSD #m #Dipole length

#Dielectric tensor values
#Calculate R,L and P
R=1-((Pi_e**2/omega**2)*(omega/(omega+Omega_e)))-((Pi_he**2/omega**2)*(omega/(omega+Omega_he))) #Right-hand polarized wave
L=1-((Pi_e**2/omega**2)*(omega/(omega-Omega_e)))-((Pi_he**2/omega**2)*(omega/(omega-Omega_he))) #Left-hand polarized wave
P=1-(Pi_e**2/omega**2)-(Pi_he**2/omega**2) #Unmagnetized plasma

#Calculate S and D
S=(R+L)/2
D=(R-L)/2

#%%
# =============================================================================
# Functions
# =============================================================================

def dipoleVacFieldBt(r,z):
	"""
	This function calculates the azimuthal component of the vacuum magnetic 
	field due to an electric dipole antenna.
	
	Equation 3.30
	
	Args:
		r,z (float): Position at which we want the component value
	
	Returns:
		Bt (float): Azimuthal magnetic field vector component
	"""
	
	#Initialize
	Bt=0
	
	#Calculate the azimuthal magnetic field component
	constTerm=mu_0*antI/(4*np.pi*r)
	geomTerm1=(z+l/2)/np.sqrt(r**2+(z+l/2)**2)
	geomTerm2=(z-l/2)/np.sqrt(r**2+(z-l/2)**2)
	
	Bt=constTerm*(geomTerm1-geomTerm2)
	
	return Bt

def dipoleVacFieldBr(r,z):
	"""
	This function calculates the radial component of the vacuum magnetic 
	field due to an electric dipole antenna.
	
	Args:
		r,z (float): Position at which we want the component value
	
	Returns:
		Br (float): Radial magnetic field vector component
	"""
	
	return 0

def hankelTransformIntegrand(r,z,kPerp,funcName):
	"""
	This function calculates the integrand for the Hankel transform
	
	NOTE: Used for integrating 'by hand'
	
	Equation 3.14
	
	Args:
		r,z (float): Position at which to calculate the Hankel transform
		kPerp (float): Perpendicular angular wavenumber
		funcName (object): Name of the function to be Hankel transformed
		
	Returns:
		The integrand of the Hankel transform of the function (funcName)
	"""
	
	return funcName(r,z)*sc.special.jv(1,kPerp*r)*r

def hankelTransform(z,kPerp,funcName):
	"""
	This function calculates the Hankel transform of the vaccum magnetic field
	of the antenna
	
	WARNING: Do not set lower limit to 0. You will get a divide by 0 error
	
	Equation 3.14

	Args:
		z (float): Z position at which to calculate the Hankel transform
		kPerp (float): Perpendicular angular wavenumber
		funcName (object): Name of the function to be Hankel transformed

	Returns:
		BHank (float): Hankel transform of the given function
	"""
	
	#Initialize
	BHank=0
	
	#Integrate using scipy
	BHank=quad(hankelTransformIntegrand,0,np.infty,args=(z,kPerp,funcName))[0]
	
	return BHank

def derivative(pos,funcName,args=(),order=1,delta=1):
	"""
	This function calculates the derivative of an arbitrary function at a given
	position. It can also do higher order derivatives
	
	Args:
		pos (float): Position at which we want the derivative of the function
		funcName (object): Function whose derivative is being calculated
		args (tuple): Any extra arguments needed for the function
		order (int): Order of the derivative
		delta (float): Distance between the 2 points for derivative 
					   calculations
	
	Returns:
		slope (float): Derivative of the function (funcName) at a given 
					   position
	"""
	
	#Base case
	if order==1:
		#Calculate the function values
		val1=funcName(pos,*args)
		val2=funcName(pos+delta,*args)
		
		slope=(val2-val1)/delta
		
		return slope
	
	#Recursive case
	else:		
		#Calculate the function values
		val1=derivative(pos,funcName,args=args,order=order-1,delta=delta)
		val2=derivative(pos+delta,funcName,args=args,order=order-1,delta=delta)
		
		slope=(val2-val1)/delta
		
		return slope

def thetaSourceTerm(z,kPerp):
	"""
	This function calculates the theta source term.
	
	Equation 3.20
	
	Args:
		z (float): Z position at which we are calculating the effect
		kPerp (float): Perpendicular angular wavenumber
		
	Returns:
		fz (float): Value of the source term at z
	"""
	
	#Normalized angular wavenumber
	nPerp=kPerp*c/omega
	
	#Components of the source term
	term1=S*((c/omega)**2)*derivative(z,hankelTransform,args=(kPerp,dipoleVacFieldBt),order=2,delta=0.001)
	term2=1j*D*((c/omega)**2)*derivative(z,hankelTransform,args=(kPerp,dipoleVacFieldBr),order=2,delta=0.001)
	term3=(R*L-S*nPerp**2)*hankelTransform(z,kPerp,dipoleVacFieldBt)
	term4=1j*D*(nPerp**2)*hankelTransform(z,kPerp,dipoleVacFieldBr)
	
	fz=-term1-term2-term3+term4
	
	return fz
	
def radialSourceTerm(z,kPerp):
	"""
	his function calculates the radial source term.
	
	Equation 3.22

	Args:
		z (float): Z position at which we are calculating the effect
		kPerp (float): Perpendicular angular wavenumber

	Returns:
		gz (float): Value of the source term at z

	"""
	
	#Normalized angular wavenumber
	nPerp=kPerp*c/omega
	
	#Components of the source term
	term1=1j*D*((c/omega)**2)*derivative(z,hankelTransform,args=(kPerp,dipoleVacFieldBt),order=2,delta=0.001)
	term2=S*((c/omega)**2)*derivative(z,hankelTransform,args=(kPerp,dipoleVacFieldBr),order=2,delta=0.001)
	term3=R*L(1-(nPerp**2)/P)*hankelTransform(z,kPerp,dipoleVacFieldBr)
	
	gz=term1-term2-term3
	
	return gz
	
def kParPlus(kPerp):
	"""
	This function calculates the forward parallel angular wavenumber
	
	Equation 3.19
	
	Args:
		kPerp (float): Perpendicular angular wavenumber
		
	Returns:
		kPlus (float): Forward parallel angular wavenumber
	"""
	
	#Normalized angular wavenumber
	nPerp=kPerp*c/omega
	
	#Components of the wavenumber
	term1=((nPerp**2)/2)*(1+S/P)
	subTerm1=(((nPerp**2)/2)**2)*(1-S/P)**2
	subTerm2=(D**2)*(1-(nPerp**2)/P)
	term2=np.sqrt(subTerm1+subTerm2)
	
	kPlusNorm=S-term1+term2
	
	kPlus=cmath.sqrt(kPlusNorm*((omega/c)**2))
	
	return kPlus

def kParMinus(kPerp):
	"""
	This function calculates the backward parallel angular wavenumber
	
	Equation 3.19
	
	Args:
		kPerp (float): Perpendicular angular wavenumber
		
	Returns:
		kMinus (float): Backward parallel angular wavenumber
	"""
	
	#Normalized angular wavenumber
	nPerp=kPerp*c/omega
	
	#Components of the wavenumber
	term1=((nPerp**2)/2)*(1+S/P)
	subTerm1=(((nPerp**2)/2)**2)*((1-S/P)**2)
	subTerm2=(D**2)*(1-(nPerp**2)/P)
	term2=np.sqrt(subTerm1+subTerm2)
	
	kMinusNorm=S-term1-term2
	
	kMinus=cmath.sqrt(kMinusNorm*((omega/c)**2))
	
	return kMinus

def bTHankel(kPerp,z):
	"""
	This function calculates the Hankel transformed theta component of the
	plasma wave field.
	
	Equation 3.35

	Args:
		kPerp (float): Perpendicular angular wavenumber
		z (float): z posiiton at which we want the field value

	Returns:
		bTheta (float): Hankel transformed field value
	"""
	
	#Initialize
	bTheta=0
	
	#Normalized angular wavenumbers
	nPerp=kPerp*c/omega
	nParPlus=kParPlus(kPerp)*c/omega
	nParMinus=kParMinus(kPerp)*c/omega
	n=np.sqrt(nPerp**2+nParPlus**2)
	
	#Components of the hankel transform
	term1=1j*mu_0*antI/(2*np.pi*kPerp)
	term2=S-R*L/(n**2)
	term3Num=(nPerp**2)*(np.e**(1j*z*kParPlus(kPerp)))*np.sin(kParPlus(kPerp)*l/2)
	term3Den=(nParPlus**2)*(nParPlus**2-nParMinus**2)
	term3=term3Num/term3Den
	
	bTheta=term1*term2*term3
	
	return bTheta

def bRHankel(kPerp,z):
	"""
	This function calculates the Hankel transformed radial component of the
	plasma wave field.
	
	Equation 3.36

	Args:
		kPerp (float): Perpendicular angular wavenumber
		z (float): z posiiton at which we want the field value

	Returns:
		bRadial (float): Hankel transformed field value
	"""
	
	#Initialize
	bRadial=0
	
	#Normalized angular wavenumbers
	nPerp=kPerp*c/omega
	nParPlus=kParPlus(kPerp)*c/omega
	nParMinus=kParMinus(kPerp)*c/omega
	n=np.sqrt(nPerp**2+nParPlus**2)
	
	#Components of the hankel transform
	term1=mu_0*antI/(2*np.pi*kPerp)
	term2Num=D*(nPerp**2)*(np.e**(1j*z*kParPlus(kPerp)))*np.sin(kParPlus(kPerp)*l/2)
	term2Den=(n**2)*(nParPlus**2-nParMinus**2)
	term2=term2Num/term2Den
	
	bRadial=term1*term2
	
	return bRadial

#%%
# =============================================================================
# Debugging
# =============================================================================
	
#Normalized kPerp array
kPerpNormArr=np.logspace(-3,2,1000)

#Not normalized kPerp array
kPerpArr=kPerpNormArr/plasmaSD

#Array to store the Hankel transformed values
bTArr=[]
bRArr=[]

#Find the Hnakel transformed arrays
for k in kPerpArr:
	bTArr.append(bTHankel(k,50))
	bRArr.append(bRHankel(k,50))
	
#Convert to numpy arrays
bTArr=np.array(bTArr)
bRArr=np.array(bRArr)

#Take the absolute value
bTAbs=np.abs(bTArr)
bRAbs=np.abs(bRArr)

#Normalize the field value arrays
bTNorm=(2*np.pi*kPerpArr/(mu_0*antI))*bTAbs
bRNorm=(2*np.pi*kPerpArr/(mu_0*antI))*bRAbs

#%%
# =============================================================================
# Plotting
# =============================================================================

plt.plot(kPerpNormArr,bTNorm,label=r'$\widetilde{B}_{\theta}$')
plt.plot(kPerpNormArr,bRNorm,label=r'$\widetilde{B}_{r}$')

plt.xlabel(r'$k_{\perp}\delta_e$')
plt.ylabel(r'$\left(\frac{2\pi k_{\perp}}{\mu_{0}I}\right )|\widetilde{B}_j|$')
plt.xlim(1e-3,1e2)
plt.ylim(1e-7,2)
plt.xscale('log')
plt.yscale('log')

plt.legend()
plt.grid(True)

plt.show()
plt.close()