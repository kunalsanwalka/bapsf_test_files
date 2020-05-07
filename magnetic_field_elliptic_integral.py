# -*- coding: utf-8 -*-

import numpy as np
import scipy.constants as const
from scipy.integrate import quad
import matplotlib.pyplot as plt

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

	############### Array with the current values for each loop ###############
    
	#0.1T constant in the LAPD
	#arr1=np.full(11,2600*10) #Yellow magnets
	#arr2=np.full(34,910*28) #Purple magnets
	#currIArr=np.concatenate([arr1,arr2,arr1]) #Magnets are arranged- Y-P-Y

	#500G from -10m to 0m and 1000G from 0m to 10m
    #Format for the current is-
    #([NUM RINGS] , [CURR PER LOOP] * [LOOPS PER RING])
	arr11=np.full(11,1300*10)
	arr21=np.full(17,465*28)
	arr22=np.full(17,930*28)
	arr12=np.full(11,2600*10)
	currIArr=np.concatenate([arr11,arr21,arr22,arr12])

	#Initialize Bx,By and Bz
	Bx=0
	By=0
	Bz=0

	#Find Bx,By and Bz due to each current loop and add the results to the final answer
	for i in range(len(zPosArr)):
        #Do not cosider current loops more than 3m away
		if np.abs(z-zPosArr[i])<3:
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

	#Terms before the integral
	currTerm=const.mu_0*currI*a/const.pi
	constTerm=1/((a+r)**2+z**2)**(3/2)

	#Calculate the integral
	integral=quad(BzIntegrand,0,np.pi/2,args=(r,z,a))

	#Find Br
	Bz=currTerm*constTerm*integral[0]

	return Bz

###########################################################
############### Generate data for the plots ###############
###########################################################

#Arrays over which to iterate
xArr=np.linspace(-1,1,50)
zArr=np.linspace(-10,10,100)
#Create the mesh
X,Z=np.meshgrid(xArr,zArr)

#Arrays with the locations of the current loops
zPosArr=np.concatenate([np.arange(-8.8,9,0.32),np.arange(-8.8,9,0.32)])
xPosArr=np.concatenate([np.full(56,1.374775/2),np.full(56,-1.374775/2)])

#Arrays to store the Bx and Bz values
BxArr=[]
BzArr=[]
#Calculate Bx and Bz
for z in zArr:
	tempBxArr=[]
	tempBzArr=[]

	for x in xArr:
		Bx,By,Bz=Bfield(x,0,z)
		tempBxArr.append(Bx)
		tempBzArr.append(Bz)

	BxArr.append(tempBxArr)
	BzArr.append(tempBzArr)

######################################################################
############### Vector plot of (Bx,Bz) on the XZ Plane ###############
######################################################################

plt.figure(figsize=(12,5))
plt.quiver(Z,X,BzArr,BxArr)
plt.scatter(zPosArr,xPosArr) #Locations of the current loops
plt.title('(Bx,Bz) on the XZ Plane')
plt.xlabel('Axial Position [m]')
plt.ylabel('Radial Position [m]')
#plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/Bx_Bz_XZ_Plane.png',dpi=600)

#Display the plot
# plt.show()

#Close the figure
plt.close()

##########################################################
############### Contour plots of Bx and Bz ###############
##########################################################

#Create the subplots
fig=plt.figure(figsize=(12,5))
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)

#Make sure they have the same levels to allow for the same colorbar
maxVal=np.max([np.max(BxArr),np.max(BzArr)])
numLevels=np.linspace(-maxVal,maxVal,100)

#Plot Bx
ax=ax1
p1=ax.contourf(X,Z,BxArr,levels=numLevels,vmax=maxVal,vmin=-maxVal)
ax.set_title('Bx')
fig.colorbar(p1,ax=ax1)

#Plot Bz
ax=ax2
p2=ax.contourf(X,Z,BzArr,levels=numLevels,vmax=maxVal,vmin=-maxVal)
ax.set_title('Bz')
fig.colorbar(p2,ax=ax2)

#Display the plot
# plt.show()

#Close the figure
plt.close()

#########################################################
############### Line plot of Bz along r=0 ###############
#########################################################

#Get By along r=0
ByArr=np.transpose(BzArr)[25]*10000

plt.plot(zArr,ByArr)
plt.title('Magnetic Field Intensity')
plt.xlabel('Axial Position [m]')
plt.ylabel('Magnetic Field Stength [G]')
plt.yticks(np.arange(0,1000+1,100))
plt.grid(True)
#plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/Bz_lineplot.png',dpi=600)

#Display the plot
plt.show()

#Close the figure
plt.close()

#Save the data in a text file
np.savetxt('near_mags.txt',np.transpose(BzArr)[25])