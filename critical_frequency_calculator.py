# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:50:16 2019

@author: kunalsanwalka

This program calculates the critical frequency for reflection based on the shape of the alfven phase velocity curve
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.integrate import quad

#Universal constants
m_amu=1.66053906660e-27 #Atomic Mass Unit
mu_0=4*np.pi*1e-7

def Bfield(x,y,z):
	"""
	This function calculates the magnetic field vector due to all the electromagnets in the LAPD

	Args:
		x,y,z: Global co-ordinate values
	Returns:
		Bx,By,Bz: Components of the magentic field vector in the global cartesian frame
	"""

	#Radius of the LAPD Magnets
	rad=1.374775/2 #meters

	#Array with the positions of the LAPD Magnets
	#Here, z=0 is the center of the chamber
	zPosArr=np.arange(-8.8,9,0.32)

	##Array with the current values for each loop
	##0.1T constant in the LAPD
	#arr1=np.full(11,2600*10) #Yellow magnets
	#arr2=np.full(34,910*28) #Purple magnets
	#currIArr=np.concatenate([arr1,arr2,arr1]) #Magnets are arranged- Y-P-Y

	#0.5T from -10m to 0m and 0.15T from 0m to 10m
	arr11=np.full(11,2600*10*0.5)
	arr21=np.full(17,910*28*0.5)
	arr22=np.full(17,910*28*1.5)
	arr12=np.full(11,2600*10*1.5)
	currIArr=np.concatenate([arr11,arr21,arr22,arr12])

	#Initialize Bx,By and Bz
	Bx=0
	By=0
	Bz=0

	#Find Bx,By and Bz due to each current loop and add the results to the final answer
	for i in range(len(zPosArr)):
		BxLoc,ByLoc,BzLoc=BFieldLocal(x,y,z-zPosArr[i],currIArr[i],rad)
		Bx+=BxLoc
		By+=ByLoc
		Bz+=BzLoc

	return Bx,By,Bz

def BFieldLocal(x,y,z,currI,rad):
	"""
	This function calculates the magentic field vector due to a current loop in local co-ordinates

	Args:
		x,y,z: Local co-ordinate values
		currI: Current in the loop
		rad: Radius of the current loop
	Returns:
		Bx,By,Bz: Components of the magnetic field vector in the local cartesian frame
	"""

	#Convert to cylindrical co-ordinates as the solver needs it in that form
	r=np.sqrt(x**2+y**2)

	if x<0 and y<0:
		r=r
	elif x<0 or y<0:
		r=-r

	#Find the fields
	Br=BrSolver(r,z,rad,currI)
	Bz=BzSolver(r,z,rad,currI)

	#Convert the fields into cartesian co-ordinates
	theta=np.pi/2
	if x!=0:
		theta=np.arctan(y/x)
	Bx=Br*np.cos(theta)
	By=Br*np.sin(theta)

	return Bx,By,Bz

def BrIntegrand(psi,r,z,a):
	"""
	This function defines the integrand for the BrSolver function

	Args:
		psi: Variable over which to integrate
		r,z: The point at which to calculate the value in cylindrical co-ordinates
		a: Radius of the current loop
	Returns:
		The integrand of the function
	"""
	
	#Numerator
	num=np.sin(psi)**2-np.cos(psi)**2

	#Notation to ease function definition
	kSq=4*a*r/((a+r)**2+z**2)

	#Denominator
	den=(1-kSq*np.sin(psi)**2)**(3/2)

	return num/den

def BrSolver(r,z,a,currI):
	"""
	This function solves for the Br component of the magnetic field in the local co-ordinate frame

	Args:
		r,z: The point at which to calculate the value in cylindrical co-ordinates
		currI: The current in the loop
		a: Radius of the current loop
	Returns:
		Br: Magnitude of the Br component of B at the given position
	"""

	#Term to ease integrand notation
	kSq=4*a*r/((a+r)**2+z**2)

	#Terms before the integral
	currTerm=const.mu_0*currI*a/const.pi
	constTerm=z/((a+r)**2+z**2)**(3/2)

	#Calculate the integral
	integral=quad(BrIntegrand,0,np.pi/2,args=(r,z,a))

	#Find Br
	Br=currTerm*constTerm*integral[0]

	return Br

def BzIntegrand(psi,r,z,a):
	"""
	This function defines the integrand for the BzSolver function

	Args:
		psi: Variable over which to integrate
		r,z: The point at which to calculate the value in cylindrical co-ordinates
		a: Radius of the current loop
	Returns:
		The integrand of the function
	"""

	#Notation to ease function definition
	kSq=4*a*r/((a+r)**2+z**2)

	#Numerator
	num=a+r-2*r*np.sin(psi)**2

	#Denominator
	den=(1-kSq*np.sin(psi)**2)**(3/2)

	return num/den

def BzSolver(r,z,a,currI):
	"""
	This function solves for the Bz component of the magnetic field in the local co-ordinate frame

	Args:
		r,z: The point at which to calculate the value in cylindrical co-ordinates
		currI: The current in the loop
		a: Radius of the current loop
	Returns:
		Bz: Magnitude of the Br component of B at the given position
	"""

	#Term to ease integrand notation
	kSq=4*a*r/((a+r)**2+z**2)

	#Terms before the integral
	currTerm=const.mu_0*currI*a/const.pi
	constTerm=1/((a+r)**2+z**2)**(3/2)

	#Calculate the integral
	integral=quad(BzIntegrand,0,np.pi/2,args=(r,z,a))

	#Find Br
	Bz=currTerm*constTerm*integral[0]

	return Bz

def Bnorm(x,y,z,forDens=False):
    """
    Calculates the magnitude of the magnetic field at a given point
    
    Args:
        x,y,z: Position in the plasma (float)
        forDens: If this function is being used for the density() function (boolean)
    Returns:
        Magnitude of the magnetic field (B) vector
    """
    if forDens==True:
        Bx,By,Bz=magneticField(x,y,z)
        return np.sqrt(Bx**2+By**2+Bz**2)
    else:
        Bx,By,Bz=Bfield(x,y,z)
        return np.sqrt(Bx**2+By**2+Bz**2)

def density(x,y,z,bMag0=0.1):
    """
    Calculates the density of the plasma at a given point
    
    Args:
        x,y,z: Position at which we need to calculate the density (float)
		bMag0: Maximum value of the magnetic field
		     : Default value is 0.1T
    Returns:
        dens: Density of the plasma at a given (x,y,z) (float)
    """
    #Equilibrium values
    a=15 #Steepness of the dropoff
    fwhm=0.3 #Full width half max

    #Base Density
    densBase=1e18
    
    #Magnetic Field magnitude
    bMag=Bnorm(x,y,z)

    #Radial Distance
    r=np.sqrt(x**2+y**2)
    
    #Mirror ratio
    mRatio=np.sqrt(bMag/bMag0)
    
    #Density
    dens=densBase*(np.tanh(-a*(mRatio*r-fwhm))+1)/2
    
    #Rescale density so that we maintain the same number of particles
    #Based on an analytic integral of the radial density function
    #Integration limit
    intLim=10
    #Numerator
    coshVal=np.cosh(a*(fwhm+intLim/mRatio))
    sechVal=1/np.cosh(a*(fwhm-intLim/mRatio))
    num=np.log(coshVal*sechVal)
    #Denominator
    coshVal=np.cosh(a*(fwhm+intLim))
    sechVal=1/np.cosh(a*(fwhm-intLim))
    den=np.log(coshVal*sechVal)
    #Rescale
    rescaleFac=num/(den*mRatio)
    dens/=rescaleFac
    
    return dens

#Array with the z-values
zValArr=np.linspace(-6,-3,2000)

#Maximum value of the magnetic field
bMag0=0
for z in zValArr:
	locBMag=Bnorm(0,0,z)
	if locBMag>bMag0:
		bMag0=locBMag

#Magnetic field strength
bMagArr=[]
#Density
densArr=[]
#Alfven wave velocity
alfvenVelArr=[]
for z in zValArr:
	bMagArr.append(Bnorm(0,0,z))
	densArr.append(density(0,0,z))
	localAlfvenVel=Bnorm(0,0,z)/np.sqrt(mu_0*density(0,0,z)*20*m_amu)
	alfvenVelArr.append(localAlfvenVel)

##########################################################################################################
##########The critical frequency formula is given in the paper- https://doi.org/10.1063/1.860452##########
##########################################################################################################

#Find the 1st and 2nd derivatives of the wave velocity
#1st derivative
vPrime=np.diff(alfvenVelArr)/(zValArr[1]-zValArr[0])
#2nd derivative
vPrimePrime=np.diff(vPrime)/(zValArr[1]-zValArr[0])

#Cut the arrays to ensure they all have the same size
alfvenVelArr=alfvenVelArr[2:]
vPrime=vPrime[1:]

#Find the critical frequency
critFreqArr=[]
for i in range(len(alfvenVelArr)):
	critFreq=0.5*np.sqrt((vPrime[i]**2)+np.abs(2*alfvenVelArr[i]*vPrimePrime[i]))
	critFreqArr.append(critFreq)

#Create the subplots
fig=plt.figure(figsize=(8,20))
ax1=fig.add_subplot(4,1,1) #Magnetic Field Strength
ax2=fig.add_subplot(4,1,2) #Density
ax3=fig.add_subplot(4,1,3) #Alfven Wave Velocity
ax4=fig.add_subplot(4,1,4) #Critical Frequency

#Plot the magnetic field strength
ax=ax1
p1=ax.plot(zValArr,bMagArr)
ax.set_title('Magnetic Field Magnitude')
ax.set_ylabel('Tesla')
ax.tick_params(labelbottom=False)
ax.ticklabel_format(axis='y',style='scientific',scilimits=(0,0))
ax.grid()

#Plot the density
ax=ax2
p2=ax.plot(zValArr,densArr)
ax.set_title('Density')
ax.set_ylabel(r'Particles per $m^3$')
ax.tick_params(labelbottom=False)
ax.ticklabel_format(axis='y',style='scientific',scilimits=(0,0))
ax.grid()

#Plot the Alfven wave velocity
ax=ax3
p3=ax.plot(zValArr[2:],alfvenVelArr)
ax.set_title('Alfven Wave Velocity')
ax.set_ylabel('Meters per second')
ax.tick_params(labelbottom=False)
ax.ticklabel_format(axis='y',style='scientific',scilimits=(0,0))
ax.grid()

#Plot the critical frequency
ax=ax4
p4=ax.plot(zValArr[2:],critFreqArr)
ax.set_title('Critical Frequency')
ax.set_ylabel('Hertz')
ax.set_xlabel('Z Position [m]')
ax.ticklabel_format(axis='y',style='scientific',scilimits=(0,0))
ax.grid()

#plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/critical_frequency.png',dpi=450)
plt.show()