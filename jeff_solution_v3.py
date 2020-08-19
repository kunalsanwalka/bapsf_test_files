# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:45:45 2020

@author: kunalsanwalka

This program solves for the Alfven Wave structure for a wave launched via an
electric dipole antenna. The solution is based on Chapter 3 of the thesis by 
Jeffrey Robertson.
"""
# =============================================================================
# Currently in the process of splitting up my code into real and imaginary
# parts since scipy can't integrate imaginary numbers
# =============================================================================

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

# =============================================================================
# Antenna parameters
# =============================================================================

antI=1 #A #Antenna current
omega=0.75*Omega_he #rad/s #Angular antenna frequency

#Loop antenna
antRad=0.04 #m Antenna radius

#Dipole antenna
dipoleL=20*plasmaSD #m #Dipole length

# =============================================================================
# Dielectric tensor values
# ==============================================================================
#Calculate R,L and P
R=1-((Pi_e**2/omega**2)*(omega/(omega+Omega_e)))-((Pi_he**2/omega**2)*(omega/(omega+Omega_he))) #Right-hand polarized wave
L=1-((Pi_e**2/omega**2)*(omega/(omega-Omega_e)))-((Pi_he**2/omega**2)*(omega/(omega-Omega_he))) #Left-hand polarized wave
P=1-(Pi_e**2/omega**2)-(Pi_he**2/omega**2) #Unmagnetized plasma

#Calculate S and D
S=(R+L)/2
D=(R-L)/2

#%%
# =============================================================================
# General Purpose Functions
# =============================================================================

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

#%%
# =============================================================================
# Antenna Field Functions
# =============================================================================
# Current loop in the XZ Plane
# =============================================================================

def BFieldXZLoop(x,y,z):
	"""
	This function calculates the magentic field vector due to a current loop in 
	local co-ordinates. The loop is located in the XZ Plane at the origin.

	Args:
		x,y,z: Local co-ordinate values
	Returns:
		Bx,By,Bz: Components of the magnetic field vector in the local cartesian frame
	"""

	#Convert to cylindrical co-ordinates as the solver needs it in that form
	r=np.sqrt(x**2+z**2)

	if x<0 and z<0:
		r=r
	elif x<0 or z<0:
		r=-r

	#Find the fields
	Br=BrSolver(r,y,antRad,antI)
	Baz=BazSolver(r,y,antRad,antI)

	#Convert the fields into cartesian co-ordinates
	theta=np.pi/2
	if x!=0:
		theta=np.arctan(z/x)
	Bz=Br*np.cos(theta)
	Bx=Br*np.sin(theta)
	
	By=Baz

	return Bx,By,Bz

def BrIntegrand(psi,r,az,a):
	"""
	This function defines the integrand for the BrSolver function

	Args:
		psi: Variable over which to integrate
		r,az: The point at which to calculate the value in cylindrical co-ordinates
		a: Radius of the current loop
	Returns:
		The integrand of the function
	"""
	
	#Numerator
	num=np.sin(psi)**2-np.cos(psi)**2

	#Notation to ease function definition
	kSq=4*a*r/((a+r)**2+az**2)

	#Denominator
	den=(1-kSq*np.sin(psi)**2)**(3/2)

	return num/den

def BrSolver(r,az,a,currI):
	"""
	This function solves for the radial component of the magnetic field in the 
	local co-ordinate frame

	Args:
		r,az: The point at which to calculate the value in cylindrical co-ordinates
		a: Radius of the current loop
	Returns:
		Br: Magnitude of the Br component of B at the given position
	"""

	#Terms before the integral
	currTerm=mu_0*currI*a/np.pi
	constTerm=az/((a+r)**2+az**2)**(3/2)

	#Calculate the integral
	integral=quad(BrIntegrand,0,np.pi/2,args=(r,az,a))

	#Find Br
	Br=currTerm*constTerm*integral[0]

	return Br

def BazIntegrand(psi,r,az,a):
	"""
	This function defines the integrand for the BzSolver function

	Args:
		psi: Variable over which to integrate
		r,az: The point at which to calculate the value in cylindrical co-ordinates
		a: Radius of the current loop
	Returns:
		The integrand of the function
	"""

	#Notation to ease function definition
	kSq=4*a*r/((a+r)**2+az**2)

	#Numerator
	num=a+r-2*r*np.sin(psi)**2

	#Denominator
	den=(1-kSq*np.sin(psi)**2)**(3/2)

	return num/den

def BazSolver(r,az,a,currI):
	"""
	This function solves for the Bz component of the magnetic field in the local co-ordinate frame

	Args:
		r,az: The point at which to calculate the value in cylindrical co-ordinates
		a: Radius of the current loop
	Returns:
		Bz: Magnitude of the Br component of B at the given position
	"""

	#Terms before the integral
	currTerm=mu_0*currI*a/np.pi
	constTerm=1/((a+r)**2+az**2)**(3/2)

	#Calculate the integral
	integral=quad(BazIntegrand,0,np.pi/2,args=(r,az,a))

	#Find Br
	Baz=currTerm*constTerm*integral[0]

	return Baz

# =============================================================================
# Dipole along z
# =============================================================================

def dipoleZ(x,y,z):
	
	#Convert to polar coordinates
	r=np.sqrt(x**2+y**2)
	theta=np.arctan2(y,x)
	
	#Get the polar field values
	Bt=dipoleVacFieldBt(r,z)
	Br=0
	Bz=0
	
	#Convert to cartesian
	#Construct the transformation matrix
	transMat=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	#Perform the coordinate transform
	bCart=np.matmul(transMat,np.array([Br,Bt]))
	
	#Get the cartesian representation
	Bx=bCart[0]
	By=bCart[1]
	
	return Bx,By,Bz

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
	geomTerm1=(z+dipoleL/2)/np.sqrt(r**2+(z+dipoleL/2)**2)
	geomTerm2=(z-dipoleL/2)/np.sqrt(r**2+(z-dipoleL/2)**2)
	
	Bt=constTerm*(geomTerm1-geomTerm2)
	
	return Bt

#%%
# =============================================================================
# Solver Functions
# =============================================================================

def Bx0IntegrandReal(x,y,z,kx,ky):
	"""
	Calculates the real part of the integrand for the vaccum field fourier transform.
	This is for the x component.
	
	Equation 3.52

	Args:
		x,y,z (float): Position.
		kx,ky (float): Angular wavenumber.

	Returns:
		integrand (complex): Integrand for the fourier transform
	"""
	
	fieldTerm=antennaFunc(x,y,z)[0]
	
	integrand=fieldTerm*(np.e**(-1j*kx*x))*(np.e**(-1j*ky*y))
	
	return np.real(integrand)

def By0IntegrandReal(x,y,z,kx,ky):
	"""
	Calculates the real part of the integrand for the vaccum field fourier transform.
	This is for the x component.
	
	Equation 3.52

	Args:
		x,y,z (float): Position.
		kx,ky (float): Angular wavenumber.

	Returns:
		integrand (complex): Integrand for the fourier transform
	"""
	
	fieldTerm=antennaFunc(x,y,z)[1]
	
	integrand=fieldTerm*(np.e**(-1j*kx*x))*(np.e**(-1j*ky*y))
	
	return np.real(integrand)

def Bz0IntegrandReal(x,y,z,kx,ky):
	"""
	Calculates the real part of the integrand for the vaccum field fourier transform.
	This is for the x component.
	
	Equation 3.52

	Args:
		x,y,z (float): Position.
		kx,ky (float): Angular wavenumber.

	Returns:
		integrand (complex): Integrand for the fourier transform
	"""
	
	fieldTerm=antennaFunc(x,y,z)[2]
	
	integrand=fieldTerm*(np.e**(-1j*kx*x))*(np.e**(-1j*ky*y))
	
	return np.real(integrand)

def Bx0IntegrandImag(x,y,z,kx,ky):
	"""
	Calculates the imaginary part of the integrand for the vaccum field fourier transform.
	This is for the x component.
	
	Equation 3.52

	Args:
		x,y,z (float): Position.
		kx,ky (float): Angular wavenumber.

	Returns:
		integrand (complex): Integrand for the fourier transform
	"""
	
	fieldTerm=antennaFunc(x,y,z)[0]
	
	integrand=fieldTerm*(np.e**(-1j*kx*x))*(np.e**(-1j*ky*y))
	
	return np.imag(integrand)

def By0IntegrandImag(x,y,z,kx,ky):
	"""
	Calculates the imaginary part of the integrand for the vaccum field fourier transform.
	This is for the x component.
	
	Equation 3.52

	Args:
		x,y,z (float): Position.
		kx,ky (float): Angular wavenumber.

	Returns:
		integrand (complex): Integrand for the fourier transform
	"""
	
	fieldTerm=antennaFunc(x,y,z)[1]
	
	integrand=fieldTerm*(np.e**(-1j*kx*x))*(np.e**(-1j*ky*y))
	
	return np.imag(integrand)

def Bz0IntegrandImag(x,y,z,kx,ky):
	"""
	Calculates the imaginary part of the integrand for the vaccum field fourier transform.
	This is for the x component.
	
	Equation 3.52

	Args:
		x,y,z (float): Position.
		kx,ky (float): Angular wavenumber.

	Returns:
		integrand (complex): Integrand for the fourier transform
	"""
	
	fieldTerm=antennaFunc(x,y,z)[2]
	
	integrand=fieldTerm*(np.e**(-1j*kx*x))*(np.e**(-1j*ky*y))
	
	return np.imag(integrand)

def Bx0FourierTransformReal(z,kx,ky):
	"""
	This function calculates the real part of the fourier transform of the x component of the 
	vaccum magnetic field
	
	Equation 3.52
	
	Args:
		kx,ky (float): Angular wavenumber
		z (float): z position at which we want the fourier transform
		
	Returns:
		fourierTrans[0] (complex): Value of the fourier transformed field
	"""
	
	#dblquad integrates the second argument 1st and the 1st argument is given
	#as a function of the second one (that's where the lambda function comes
	#from)
	
	fourierTrans=dblquad(Bx0IntegrandReal,-np.infty,np.infty,lambda y:-np.infty,lambda y:np.infty,args=(z,kx,ky))
	
	return fourierTrans[0]

def By0FourierTransformReal(z,kx,ky):
	"""
	This function calculates the real part of the fourier transform of the x component of the 
	vaccum magnetic field
	
	Equation 3.52
	
	Args:
		kx,ky (float): Angular wavenumber
		z (float): z position at which we want the fourier transform
		
	Returns:
		fourierTrans[0] (complex): Value of the fourier transformed field
	"""
	
	#dblquad integrates the second argument 1st and the 1st argument is given
	#as a function of the second one (that's where the lambda function comes
	#from)
	
	fourierTrans=dblquad(By0IntegrandReal,-np.infty,np.infty,lambda y:-np.infty,lambda y:np.infty,args=(z,kx,ky))
	
	return fourierTrans[0]

def Bz0FourierTransformReal(z,kx,ky):
	"""
	This function calculates the real part of the fourier transform of the x component of the 
	vaccum magnetic field
	
	Equation 3.52
	
	Args:
		kx,ky (float): Angular wavenumber
		z (float): z position at which we want the fourier transform
		
	Returns:
		fourierTrans[0] (complex): Value of the fourier transformed field
	"""
	
	#dblquad integrates the second argument 1st and the 1st argument is given
	#as a function of the second one (that's where the lambda function comes
	#from)
	
	fourierTrans=dblquad(Bz0IntegrandReal,-np.infty,np.infty,lambda y:-np.infty,lambda y:np.infty,args=(z,kx,ky))
	
	return fourierTrans[0]

def Bx0FourierTransformImag(z,kx,ky):
	"""
	This function calculates the imaginary part of the fourier transform of the x component of the 
	vaccum magnetic field
	
	Equation 3.52
	
	Args:
		kx,ky (float): Angular wavenumber
		z (float): z position at which we want the fourier transform
		
	Returns:
		fourierTrans[0] (complex): Value of the fourier transformed field
	"""
	
	#dblquad integrates the second argument 1st and the 1st argument is given
	#as a function of the second one (that's where the lambda function comes
	#from)
	
	fourierTrans=dblquad(Bx0IntegrandImag,-np.infty,np.infty,lambda y:-np.infty,lambda y:np.infty,args=(z,kx,ky))
	
	return fourierTrans[0]

def By0FourierTransformImag(z,kx,ky):
	"""
	This function calculates the imaginary part of the fourier transform of the x component of the 
	vaccum magnetic field
	
	Equation 3.52
	
	Args:
		kx,ky (float): Angular wavenumber
		z (float): z position at which we want the fourier transform
		
	Returns:
		fourierTrans[0] (complex): Value of the fourier transformed field
	"""
	
	#dblquad integrates the second argument 1st and the 1st argument is given
	#as a function of the second one (that's where the lambda function comes
	#from)
	
	fourierTrans=dblquad(By0IntegrandImag,-np.infty,np.infty,lambda y:-np.infty,lambda y:np.infty,args=(z,kx,ky))
	
	return fourierTrans[0]

def Bz0FourierTransformImag(z,kx,ky):
	"""
	This function calculates the imaginary part of the fourier transform of the x component of the 
	vaccum magnetic field
	
	Equation 3.52
	
	Args:
		kx,ky (float): Angular wavenumber
		z (float): z position at which we want the fourier transform
		
	Returns:
		fourierTrans[0] (complex): Value of the fourier transformed field
	"""
	
	#dblquad integrates the second argument 1st and the 1st argument is given
	#as a function of the second one (that's where the lambda function comes
	#from)
	
	fourierTrans=dblquad(Bz0IntegrandImag,-np.infty,np.infty,lambda y:-np.infty,lambda y:np.infty,args=(z,kx,ky))
	
	return fourierTrans[0]

def Bx0FourierTransform(z,kx,ky):
	"""
	This function calculates the fourier transform of the vaccum magnetic field
	
	Equation 3.52
	
	Args:
		kx,ky (float): Angular wavenumber
		z (float): z position at which we want the fourier transform
		
	Returns:
		fourierTrans[0] (complex): Value of the fourier transformed field
	"""
	
	realPart=Bx0FourierTransformReal(z, kx, ky)
	imagPart=Bx0FourierTransformImag(z, kx, ky)
	
	fourierTrans=realPart+1j*imagPart
	
	return fourierTrans

def kParPlus(kx,ky):
	"""
	This function calculates the forward parallel angular wavenumber
	
	Equation 3.60
	
	Args:
		kx,ky (float): Angular wavenumber
		
	Returns:
		kPlus (float): Forward parallel angular wavenumber
	"""
	
	#Normalized angular wavenumber
	nx=kx*c/omega
	ny=ky*c/omega
	nPerp=np.sqrt(nx**2+ny**2)
	
	#Components of the wavenumber
	term1=((nPerp**2)/2)*(1+S/P)
	subTerm1=((nPerp**4)/4)*((1-S/P)**2)
	subTerm2=(D**2)*(1-(nPerp**2)/P)
	term2=np.sqrt(subTerm1+subTerm2)
	
	kPlusNorm=S-term1+term2
	
	kPlus=cmath.sqrt(kPlusNorm*((omega/c)**2))
	
	return kPlus

def kParMinus(kx,ky):
	"""
	This function calculates the backward parallel angular wavenumber
	
	Equation 3.60
	
	Args:
		kx,ky (float): Angular wavenumber
		
	Returns:
		kMinus (float): Backward parallel angular wavenumber
	"""
	
	#Normalized angular wavenumber
	nx=kx*c/omega
	ny=ky*c/omega
	nPerp=np.sqrt(nx**2+ny**2)
	
	#Components of the wavenumber
	term1=((nPerp**2)/2)*(1+S/P)
	subTerm1=((nPerp**4)/4)*((1-S/P)**2)
	subTerm2=(D**2)*(1-(nPerp**2)/P)
	term2=np.sqrt(subTerm1+subTerm2)
	
	kMinusNorm=S-term1-term2
	
	kMinus=cmath.sqrt(kMinusNorm*((omega/c)**2))
	
	return kMinus

def f_z(z,kx,ky):
	"""
	This function calculates the source term f(z) for the Bx component.
	
	Equation 3.61

	Args:
		z (float): Z position
		kx,ky (float): Angular wavenumber

	Returns:
		fz (complex): Source term
	"""
	
	#Normalize the wavenumbers
	nx=kx*c/omega
	ny=ky*c/omega
	
	#Double derivatives of the components
	BxDeriv=derivative(z,Bx0FourierTransform,args=(kx,ky),order=2,delta=0.001)
	ByDeriv=derivative(z,By0FourierTransform,args=(kx,ky),order=2,delta=0.001)
	#Normalize
	BxDerivNorm=((omega/c)**2)*BxDeriv
	ByDerivNorm=((omega/c)**2)*ByDeriv
	
	#1st 2 terms of f_z
	term1=S*BxDerivNorm
	term2=1j*D*ByDerivNorm
	
	#Sub terms for the last 2 terms
	subTerm31=1j*ny*(1j*ny*S+nx*D)
	subTerm32=R*L*(1-(nx**2)/P)
	subTerm41=1j*ny*(1j*nx*S-ny*D)
	subTerm42=nx*ny*R*L/P
	
	#Last 2 terms of f_z
	term3=Bx0FourierTransform(z,kx,ky)*(-subTerm31-subTerm32)
	term4=By0FourierTransform(z,kx,ky)*(subTerm41+subTerm42)
	
	#Calculate f_z
	fz=-term1+term2+term3+term4
	
	return fz

def g_z(z,kx,ky):
	"""
	This function calculates the source term g(z) for the By component.
	
	Equation 3.62

	Args:
		z (float): Z position
		kx,ky (float): Angular wavenumber

	Returns:
		gz (complex): Source term
	"""
	
	#Normalize the wavenumbers
	nx=kx*c/omega
	ny=ky*c/omega
	
	#Double derivatives of the components
	BxDeriv=derivative(z,Bx0FourierTransform,args=(kx,ky),order=2,delta=0.001)
	ByDeriv=derivative(z,By0FourierTransform,args=(kx,ky),order=2,delta=0.001)
	#Normalize
	BxDerivNorm=((omega/c)**2)*BxDeriv
	ByDerivNorm=((omega/c)**2)*ByDeriv
	
	#1st 2 terms of f_z
	term1=1j*D*BxDerivNorm
	term2=S*ByDerivNorm
	
	#Sub terms for the last 2 terms
	subTerm31=1j*nx*(1j*ny*S+nx*D)
	subTerm32=nx*ny*R*L/P
	subTerm41=1j*nx*(1j*nx*S-ny*D)
	subTerm42=R*L*(1-(ny**2)/P)
	
	#Last 2 terms of f_z
	term3=Bx0FourierTransform(z,kx,ky)*(subTerm31+subTerm32)
	term4=By0FourierTransform(z,kx,ky)*(-subTerm41-subTerm42)
	
	#Calculate f_z
	gz=-term1-term2+term3+term4
	
	return gz

def BxIntegrand1(zPrime,z,kx,ky):
	"""
	This function calculates the 1st integrand for the Bx component of the wave
	
	Equation 3.63
	
	Args:
		zPrime (float): Variable being integrated over
		z (float): z position
		kx,ky (float): Angular wavenumber
		
	Returns:
		integrand (float): 1st integrand for Bx
	"""
	
	#Get the parallel angular wavenumber
	kPlus=kParPlus(kx,ky)
	kMinus=kParMinus(kx,ky)
	
	#Get the source term
	fz=f_z(zPrime,kx,ky)
	
	#Construct the numerators
	num1=1j*np.e**(1j*kPlus*(z-zPrime))
	num2=1j*np.e**(1j*kMinus*(z-zPrime))
	
	#Construct the denominators
	den1=2*kPlus*(kPlus**2-kMinus**2)
	den2=2*kMinus*(kMinus**2-kPlus**2)
	
	#Construct the 2 terms
	term1=num1/den1
	term2=num2/den2
	
	#Construct the integrand
	integrand=(term1+term2)*fz
	
	return integrand

def BxIntegrand2(zPrime,z,kx,ky):
	"""
	This function calculates the 2nd integrand for the Bx component of the wave
	
	Equation 3.63
	
	Args:
		zPrime (float): Variable being integrated over
		z (float): z position
		kx,ky (float): Angular wavenumber
		
	Returns:
		integrand (float): 2nd integrand for Bx
	"""
	
	#Get the parallel angular wavenumber
	kPlus=kParPlus(kx,ky)
	kMinus=kParMinus(kx,ky)
	
	#Get the source term
	fz=f_z(zPrime,kx,ky)
	
	#Construct the numerators
	num1=1j*np.e**(-1j*kPlus*(z-zPrime))
	num2=1j*np.e**(-1j*kMinus*(z-zPrime))
	
	#Construct the denominators
	den1=2*kPlus*(kPlus**2-kMinus**2)
	den2=2*kMinus*(kMinus**2-kPlus**2)
	
	#Construct the 2 terms
	term1=num1/den1
	term2=num2/den2
	
	#Construct the integrand
	integrand=(term1+term2)*fz
	
	return integrand

def ByIntegrand1(zPrime,z,kx,ky):
	"""
	This function calculates the 1st integrand for the By component of the wave
	
	Equation 3.63
	
	Args:
		zPrime (float): Variable being integrated over
		z (float): z position
		kx,ky (float): Angular wavenumber
		
	Returns:
		integrand (float): 2nd integrand for By
	"""
	
	#Get the parallel angular wavenumber
	kPlus=kParPlus(kx,ky)
	kMinus=kParMinus(kx,ky)
	
	#Get the source term
	gz=g_z(zPrime,kx,ky)
	
	#Construct the numerators
	num1=1j*np.e**(1j*kPlus*(z-zPrime))
	num2=1j*np.e**(1j*kMinus*(z-zPrime))
	
	#Construct the denominators
	den1=2*kPlus*(kPlus**2-kMinus**2)
	den2=2*kMinus*(kMinus**2-kPlus**2)
	
	#Construct the 2 terms
	term1=num1/den1
	term2=num2/den2
	
	#Construct the integrand
	integrand=(term1+term2)*gz
	
	return integrand

def ByIntegrand2(zPrime,z,kx,ky):
	"""
	This function calculates the 2nd integrand for the By component of the wave
	
	Equation 3.63
	
	Args:
		zPrime (float): Variable being integrated over
		z (float): z position
		kx,ky (float): Angular wavenumber
		
	Returns:
		integrand (float): 2nd integrand for By
	"""
	
	#Get the parallel angular wavenumber
	kPlus=kParPlus(kx,ky)
	kMinus=kParMinus(kx,ky)
	
	#Get the source term
	gz=g_z(zPrime,kx,ky)
	
	#Construct the numerators
	num1=1j*np.e**(-1j*kPlus*(z-zPrime))
	num2=1j*np.e**(-1j*kMinus*(z-zPrime))
	
	#Construct the denominators
	den1=2*kPlus*(kPlus**2-kMinus**2)
	den2=2*kMinus*(kMinus**2-kPlus**2)
	
	#Construct the 2 terms
	term1=num1/den1
	term2=num2/den2
	
	#Construct the integrand
	integrand=(term1+term2)*gz
	
	return integrand

def BxFourierTransform(z,kx,ky):
	"""
	This function calculates the fourier transformed x component of the
	magnetic field.

	Args:
		z (float): z position
		kx,ky (float): Angular wavenumber

	Returns:
		Bx (complex): Fourier transformed wave field
	"""
	
	#Compute the integrals
	term1=quad(BxIntegrand1,-np.infty,z,args=(z,kx,ky))[0]
	term2=quad(BxIntegrand2,z,np.infty,args=(z,kx,ky))[0]
	
	#Normalize
	normTerm1=((omega/c)**4)*term1
	normTerm2=((omega/c)**4)*term2
	
	#Calculate the fourier transformed wave field
	Bx=normTerm1+normTerm2
	
	return Bx

def ByFourierTransform(z,kx,ky):
	"""
	This function calculates the fourier transformed y component of the
	magnetic field.

	Args:
		z (float): z position
		kx,ky (float): Angular wavenumber

	Returns:
		By (complex): Fourier transformed wave field
	"""
	
	#Compute the integrals
	term1=quad(ByIntegrand1,-np.infty,z,args=(z,kx,ky))[0]
	term2=quad(ByIntegrand2,z,np.infty,args=(z,kx,ky))[0]
	
	#Normalize
	normTerm1=((omega/c)**4)*term1
	normTerm2=((omega/c)**4)*term2
	
	#Calculate the fourier transformed wave field
	By=normTerm1+normTerm2
	
	return By

#%%
# =============================================================================
# Debugging
# =============================================================================
	
#Antenna type
antennaFunc=dipoleZ

#Normalized kPerp array
kPerpNormArr=np.logspace(-3,2,1000)

#Not normalized kPerp array
kPerpArr=kPerpNormArr/plasmaSD

#Arrays to store the fourier transformed values
BtArr=[]
BrArr=[]

for k in tqdm(kPerpArr,position=0):
	#Get the cartesian representation
	Bx=BxFourierTransform(50,k,0)
	By=ByFourierTransform(50,k,0)
	
	#Convert to polar coordinates
	kr=k
	ktheta=0
	#Construct the transformation matrix
	transMat=np.array([[np.cos(ktheta),np.sin(ktheta)],[-np.sin(ktheta),np.cos(ktheta)]])
	#Perform the coordinate transform
	bPolar=np.matmul(transMat,np.array([Bx,By]))
	
	Br=bPolar[0]
	Bt=bPolar[1]
	
	BrArr.append(Br)
	BtArr.append(Bt)
	
#Convert to numpy arrays
bTArr=np.array(BtArr)
bRArr=np.array(BrArr)

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