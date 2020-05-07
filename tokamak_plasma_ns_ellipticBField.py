import numpy as np
import scipy.constants as const
from scipy.integrate import quad
from scipy.linalg import inv
#from petram.helper.variables import variable

#Variables imported from the model description
freq=275*1000 #Hz

#Equilibrium Values (help define the density function)
fwhm=0.3 #Full Width Half Max
bMag0=0.05 #Magnetic Field Strength
a=15 #Steepness of the dropoff

#Simulation Constants  
w=freq*2*np.pi #Angular frequency of the antenna

#Universal Constants
eps_0=8.85418782e-12 #Vacuum Permittivity
q_e=-1.60217662e-19 #Electron Charge
q_p=1.60217662e-19 #Proton Charge
m_e=9.10938356e-31 #Electron Mass
m_amu=1.66053906660e-27 #Atomic Mass Unit

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

	#Array with the current values for each loop
	#0.1T constant in the LAPD
	arr1=np.full(11,2600*10) #Yellow magnets
	arr2=np.full(34,910*28) #Purple magnets
	currIArr=np.concatenate([arr1,arr2,arr1]) #Magnets are arranged- Y-P-Y

	#0.1T from -10m to 0m and 0.2T from 0m to 10m
	#arr11=np.full(11,2600*10)
	#arr21=np.full(17,910*28)
	#arr22=np.full(17,910*28*2)
	#arr12=np.full(11,2600*10*2)
	#currIArr=np.concatenate([arr11,arr21,arr22,arr12])

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
	theta=np.pi/2 #If x=0
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

def magneticField(x,y,z):
    """
    Calculates the magnetic field vector at a given point.
    Only used for the density() function as that way, we can treat the density and B field strength as independent variables.
    
    Args:
        x,y,z: Position in the plasma (float)
    Returns:
        B: Magnetic Field Vector (array)
    """
    B=[0,0,0]
    
    #Scratch field
    #Goes from 500G to 1600G over 2m
    B[2]=(0.11/2)*(np.tanh(z)+1)+0.05
    
    return B

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

def densityConst(x,y,z):
    """
    Calculates the density of the plasma at a given point
    Note: There is no axial variation in this density function
          This function was created for debugging purposes
    
    Args:
        x,y,z: Position at which we need to calculate the density (float)
    Returns:
        dens: Density of the plasma at a given (x,y,z) (float)
    """
    #Base Case
    dens=0

    #Base Density
    densBase=1e18

    #Radial Distance
    r=np.sqrt(x**2+y**2)

    #Density
    dens=densBase*(np.tanh(-a*(r-fwhm))+1)/2

    return dens

def density(x,y,z):
    """
    Calculates the density of the plasma at a given point
    
    Args:
        x,y,z: Position at which we need to calculate the density (float)
    Returns:
        dens: Density of the plasma at a given (x,y,z) (float)
    """
    #Base Case
    dens=0
    
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

def collisionality(x,y,z):
    """
    This function calculates the collisionality as a function of the position in the plasma

    Args:
        x,y,z: Position in the plasma
    Returns:
        col: Collisionality (float)
    """    
    #Initialize col
    col=0*1e6 #MHz #Base case collisionality

#    #col is uniform through most of the plasma
#    if z<=9 and z>=-9:
#        return col
#
#    #Ramp the collisionality at the end to make sure all the wave energy is absorbed before it hits the walls
#    elif z>9:
#        #Quadratic ramp for the collisionality
#        zLoc=z-9 #Work in local co-ordinates
#        col+=1e8*(zLoc**2) #Ramps to 100MHz
#    elif z<-9:
#        #Quadratic ramp for the collisionality
#        zLoc=-z-9 #Work in local co-ordinates
#        col+=1e8*(zLoc**2) #Ramps to 100MHz

    return col

#@variable.array(complex = True, shape=(3,3))
def epsilonr_plasma(x,y,z):
    """
    Calculates the plasma di-electric tensor at a given point in the plasma.
    Based on the cold plasma approximation.
    
    Args:
        x,y,z: Position in the plasma (float)
    Returns:
        epsR: The plasma di-electric tensor at the given point (3x3 Matrix)
    """
    #Get the magnetic field
    Bx,By,Bz=Bfield(x,y,z)

    #Get the relevant angles for matrix rotation later
    Bmag=Bnorm(x,y,z)
    theta=np.arccos(Bz/Bmag)
    phi=np.pi/2
    if Bx!=0:
        phi=np.arctan(By/Bx)

    #Get the density of the plasma
    ne=density(x,y,z)

    #Get the collisionality of the plasma
    nu_e=collisionality(x,y,z)

    #If the density is too low, the tensor can just be the identity matrix
    if ne < 1e5:
        epsR=np.array([[1,0,0],[0,1,0],[0,0,1]])
        #MFEM uses column major for matrix
        return np.conj(epsR)

    #Calcuate the relevant frequencies
    Pi_he=np.sqrt(ne*q_p**2/(eps_0*4*m_amu)) #Helium plasma frequency
    Pi_e=np.sqrt(ne*q_e**2/(eps_0*m_e)) #Electron plasma frequency
    Omega_he=q_p*Bmag/(4*m_amu) #Helium cyclotron frequency
    Omega_e=q_e*Bmag/(m_e) #Electron cyclotron frequency
   
    #Calculate R,L and P
    R=1-((Pi_e**2/w**2)*(w/(w+Omega_e)))-((Pi_he**2/w**2)*(w/(w+Omega_he))) #Right-hand polarized wave
    L=1-((Pi_e**2/w**2)*(w/(w-Omega_e)))-((Pi_he**2/w**2)*(w/(w-Omega_he))) #Left-hand polarized wave
    P=1-(Pi_e**2/(w*(w-1j*nu_e)))-(Pi_he**2/w**2) #Unmagnetized plasma

    #Calculate S and D
    S=(R+L)/2
    D=(R-L)/2

    #Construct the di-electric tensor elements
    e_xx=S
    e_xy=-1j*D
    e_xz=0

    e_yx=1j*D
    e_yy=S
    e_yz=0

    e_zx=0
    e_zy=0
    e_zz=P

    #Construct the eps_r matrix in the local frame
    epsRLoc=np.array([[e_xx,e_xy,e_xz],[e_yx,e_yy,e_yz],[e_zx,e_zy,e_zz]])

    #Construct the rotation matrices
    #Phi
    phiMat=np.array([[np.cos(phi),np.sin(phi),0],[-np.sin(phi),np.cos(phi),0],[0,0,1]])
    #Theta
    thetaMat=np.array([[np.cos(theta),0,-np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]])

    #Apply the rotation matrices
    #Apply the phi rotation matrix
    temp=np.matmul(phiMat,epsRLoc)
    epsRtemp=np.matmul(temp,np.linalg.inv(phiMat))
    #Apply the theta rotation matrix
    temp=np.matmul(thetaMat,epsRtemp)
    epsR=np.matmul(temp,np.linalg.inv(thetaMat))

    #MFEM uses column major for matrix
    return epsR

def PML(r,co_ord):
    """
    This function find the value of S_r, the PML co-ordinate stretch that allows for damping.

    Args:
        r: Value of the unstretched co-ordinate (float)
        co_ord: Name of the co-ordinate being stretched (string)
    Returns:
        S_r: Stretched co-ordinate
    """
    #We are only stretching the z-co-ordinate for now
    if co_ord=='z':
        #PML is from -2.5 to -0.5 and from 3 to 5

        #Initialize S_r
        S_r=1
        
        if r>0:
            #Postive Region PML

            #Take the absolute value of r
            rAbs=np.abs(r)

            #Define L_r and L_PML (define where the PML begins and its depth)
            PMLbegin=0.5#m (LAPD is 20m long)
            PMLdepth=1#m (PML Layer is 2m long)

            #Define S_r' and S_r'' (the real and complex stretch respectively)
            realStretch=(25*(rAbs-PMLbegin))**5 #Damps evanescent waves
            complexStretch=(25*(rAbs-PMLbegin))**5 #Damps real waves

            #Define p_r (ramping factor)
            rampFact=2

            #Calculate S_r
            if rAbs>PMLbegin:
                S_r=1+(realStretch+1j*complexStretch)*(((rAbs-PMLbegin)/PMLdepth)**rampFact)

        elif r<=0:
            #Negative Region PML

            #Take the absolute value of r
            rAbs=np.abs(r)

            #Define L_r and L_PML (define where the PML begins and its depth)
            PMLbegin=3#m (LAPD is 20m long)
            PMLdepth=1#m (PML Layer is 2m long)

            #Define S_r' and S_r'' (the real and complex stretch respectively)
            realStretch=(25*(rAbs-PMLbegin))**5 #Damps evanescent waves
            complexStretch=(25*(rAbs-PMLbegin))**5 #Damps real waves

            #Define p_r (ramping factor)
            rampFact=2

            #Calculate S_r
            if rAbs>PMLbegin:
                S_r=1+(realStretch+1j*complexStretch)*(((rAbs-PMLbegin)/PMLdepth)**rampFact)
        return S_r
    else:
        return 1

#@variable.array(complex = True, shape=(3,3))
def epsilonr_pml(x,y,z):
    """
    Used only for the PML region.
    Calculates the plasma di-electric tensor at a given point in the plasma.
    Based on the cold plasma approximation.
    
    Args:
        x,y,z: Position in the plasma (float)
    Returns:
        epsR: The plasma di-electric tensor at the given point (3x3 Matrix)
    """
    #Get the magnetic field
    Bx,By,Bz=Bfield(x,y,z)

    #Get the relevant angles for matrix rotation later
    Bmag=Bnorm(x,y,z)
    theta=np.arccos(Bz/Bmag)
    phi=np.pi/2
    if Bx!=0:
        phi=np.arctan(By/Bx)

    #Get the density of the plasma
    ne=density(x,y,z)

    #Get the collisionality of the plasma
    nu_e=collisionality(x,y,z)

    #If the density is too low, the tensor can just be the identity matrix
    if ne < 1e5:
        epsR=np.array([[1,0,0],[0,1,0],[0,0,1]])
        #MFEM uses column major for matrix
        return np.conj(epsR)

    #Calcuate the relevant frequencies
    Pi_he=np.sqrt(ne*q_p**2/(eps_0*4*m_amu)) #Helium plasma frequency
    Pi_e=np.sqrt(ne*q_e**2/(eps_0*m_e)) #Electron plasma frequency
    Omega_he=q_p*Bmag/(4*m_amu) #Helium cyclotron frequency
    Omega_e=q_e*Bmag/(m_e) #Electron cyclotron frequency
   
    #Calculate R,L and P
    R=1-((Pi_e**2/w**2)*(w/(w+Omega_e)))-((Pi_he**2/w**2)*(w/(w+Omega_he))) #Right-hand polarized wave
    L=1-((Pi_e**2/w**2)*(w/(w-Omega_e)))-((Pi_he**2/w**2)*(w/(w-Omega_he))) #Left-hand polarized wave
    P=1-(Pi_e**2/(w*(w-1j*nu_e)))-(Pi_he**2/w**2) #Unmagnetized plasma

    #Calculate S and D
    S=(R+L)/2
    D=(R-L)/2

    #Construct the di-electric tensor elements
    #PML Layer is implemented at the following z-values-
    #-2.5<z<-0.5 and 3<z<5
    #It is also only in the z-direction as that is the direction of propagation of the Alfven wave
    e_xx=S*(PML(y,'y')*PML(z,'z')/PML(x,'x'))
    e_xy=-1j*D*PML(z,'z')
    e_xz=0*PML(y,'y')

    e_yx=1j*D*PML(z,'z')
    e_yy=S*(PML(z,'z')*PML(x,'x')/PML(y,'y'))
    e_yz=0*PML(x,'x')

    e_zx=0*PML(y,'y')
    e_zy=0*PML(x,'x')
    e_zz=P*(PML(x,'x')*PML(y,'y')/PML(z,'z'))

    #Construct the eps_r matrix in the local frame
    epsRLoc=np.array([[e_xx,e_xy,e_xz],[e_yx,e_yy,e_yz],[e_zx,e_zy,e_zz]])

    #Construct the rotation matrices
    #Phi
    phiMat=np.array([[np.cos(phi),np.sin(phi),0],[-np.sin(phi),np.cos(phi),0],[0,0,1]])
    #Theta
    thetaMat=np.array([[np.cos(theta),0,-np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]])

    #Apply the rotation matrices
    #Apply the phi rotation matrix
    temp=np.matmul(phiMat,epsRLoc)
    epsRtemp=np.matmul(temp,np.linalg.inv(phiMat))
    #Apply the theta rotation matrix
    temp=np.matmul(thetaMat,epsRtemp)
    epsR=np.matmul(temp,np.linalg.inv(thetaMat))

    #MFEM uses column major for matrix
    return epsR

#@variable.array(complex = True, shape=(3,3))
def mur_pml(x,y,z):
    """
    Used only for the PML region.
    Calculates the magnetic permeability tensor at a given point in the plasma.
    Based on the cold plasma approximation.
    
    Args:
        x,y,z: Position in the plasma (float)
    Returns:
        epsR: The plasma di-electric tensor at the given point (3x3 Matrix)
    """
    #Get the magnetic field
    Bx,By,Bz=Bfield(x,y,z)

    #Get the relevant angles for matrix rotation later
    Bmag=Bnorm(x,y,z)
    theta=np.arccos(Bz/Bmag)
    phi=np.pi/2
    if Bx!=0:
        phi=np.arctan(By/Bx)

    #Construct the tensor elements
    u_xx=1*(PML(y,'y')*PML(z,'z')/PML(x,'x'))
    u_xy=0
    u_xz=0

    u_yx=0
    u_yy=1*(PML(z,'z')*PML(x,'x')/PML(y,'y'))
    u_yz=0

    u_zx=0
    u_zy=0
    u_zz=1*(PML(x,'x')*PML(y,'y')/PML(z,'z'))

    #Construct the matrix
    muRLoc=np.array([[u_xx,u_xy,u_xz],[u_yx,u_yy,u_yz],[u_zx,u_zy,u_zz]])

    #Construct the rotation matrices
    #Phi
    phiMat=np.array([[np.cos(phi),np.sin(phi),0],[-np.sin(phi),np.cos(phi),0],[0,0,1]])
    #Theta
    thetaMat=np.array([[np.cos(theta),0,-np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]])

    #Apply the rotation matrices
    #Apply the phi rotation matrix
    temp=np.matmul(phiMat,muRLoc)
    muRtemp=np.matmul(temp,np.linalg.inv(phiMat))
    #Apply the theta rotation matrix
    temp=np.matmul(thetaMat,muRtemp)
    muR=np.matmul(temp,np.linalg.inv(thetaMat))

    #MFEM uses column major for the matrix
    return muR

#Check to see if any of the matrix functions return a singular matrix

#Define simulation space
xArr=np.linspace(-0.5,0.5,11)
yArr=np.linspace(-0.5,0.5,11)
zArr=np.linspace(-4,1.5,56)

for x in xArr:
    print(x)
    for y in yArr:
        for z in zArr:
            mat=mur_pml(x,y,z)
            #Check if the matrix is singular
            if np.linalg.det(mat)==0:

                #Print the position
                print('(x,y,z)=('+str(x)+','+str(y)+','+str(z)+')')

                #Print the magnetic field at the position
                Bx,By,Bz=Bfield(x,y,z)
                print('B=('+str(Bx)+','+str(By)+','+str(Bz)+')')

                #Print the PML at the position
                PMLval=PML(z,'z')
                print('PML z='+str(PMLval))

                print('----------------------------------------------------')