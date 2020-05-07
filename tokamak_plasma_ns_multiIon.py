import numpy as np
from scipy.linalg import inv
from petram.helper.variables import variable

#Variables imported from the model description
#freq: Frequency of the antenna (fundamental frequency at which the simulation is being solved

#Ratio of He-Ne in the plasma (make sure they add up to 1)
heRatio=1.0 #Helium
neRatio=0.0 #Neon

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
    Calculates the magnetic field vector at a given point

    Args:
        x,y,z: Position in the plasma (float)
    Returns:
        B: Magnetic Field Vector (array)
    """
    #Current field is a constant 1kG along the length of the chamber
    B=[0,0,0.1] #Calculations are done in Tesla
    return B

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
    densBase=1.4e18

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
    col=4.75*1e6 #MHz #Base case collisionality

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

@variable.array(complex = True, shape=(3,3))
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
    ne=densityConst(x,y,z)

    #Get the collisionality of the plasma
    nu_e=collisionality(x,y,z)

    #Calcuate the relevant frequencies
    Pi_he=np.sqrt((ne*heRatio)*q_p**2/(eps_0*4*m_amu)) #Helium plasma frequency
    Pi_ne=np.sqrt((ne*neRatio)*q_p**2/(eps_0*20*m_amu)) #Neon plasma frequency
    Pi_e=np.sqrt(ne*q_e**2/(eps_0*m_e)) #Electron plasma frequency
    Omega_he=q_p*Bmag/(4*m_amu) #Helium cyclotron frequency
    Omega_ne=q_p*Bmag/(20*m_amu) #Neon cyclotron frequency
    Omega_e=q_e*Bmag/(m_e) #Electron cyclotron frequency
   
    #Calculate R,L and P
    R=1-((Pi_e**2/w**2)*(w/(w+Omega_e)))-((Pi_he**2/w**2)*(w/(w+Omega_he)))-((Pi_ne**2/w**2)*(w/(w+Omega_ne))) #Right-hand polarized wave
    L=1-((Pi_e**2/w**2)*(w/(w-Omega_e)))-((Pi_he**2/w**2)*(w/(w-Omega_he)))-((Pi_ne**2/w**2)*(w/(w-Omega_ne))) #Left-hand polarized wave
    P=1-(Pi_e**2/(w*(w+1j*nu_e)))-(Pi_he**2/w**2)-(Pi_ne**2/w**2) #Unmagnetized plasma

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
        #PML is from -5 to -3 and from 3 to 5

        #Initialize S_r
        S_r=1
        
        if r>0:
            #Postive Region PML

            #Take the absolute value of r
            rAbs=np.abs(r)

            #Define L_r and L_PML (define where the PML begins and its depth)
            PMLbegin=3#m (LAPD is 6m long)
            PMLdepth=2#m (PML Layer is 2m long)

            #Define S_r' and S_r'' (the real and complex stretch respectively)
            realStretch=(2*(rAbs-PMLbegin))**1 #Damps evanescent waves
            complexStretch=(2*(rAbs-PMLbegin))**1 #Damps real waves

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
            PMLbegin=3#m (LAPD is 6m long)
            PMLdepth=2#m (PML Layer is 2m long)

            #Define S_r' and S_r'' (the real and complex stretch respectively)
            realStretch=(2*(rAbs-PMLbegin))**1 #Damps evanescent waves
            complexStretch=(2*(rAbs-PMLbegin))**1 #Damps real waves

            #Define p_r (ramping factor)
            rampFact=2

            #Calculate S_r
            if rAbs>PMLbegin:
                S_r=1+(realStretch+1j*complexStretch)*(((rAbs-PMLbegin)/PMLdepth)**rampFact)
        return S_r
    else:
        return 1

@variable.array(complex = True, shape=(3,3))
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
    ne=densityConst(x,y,z)

    #Get the collisionality of the plasma
    nu_e=collisionality(x,y,z)

    #Calcuate the relevant frequencies
    Pi_he=np.sqrt((ne*heRatio)*q_p**2/(eps_0*4*m_amu)) #Helium plasma frequency
    Pi_ne=np.sqrt((ne*neRatio)*q_p**2/(eps_0*20*m_amu)) #Neon plasma frequency
    Pi_e=np.sqrt(ne*q_e**2/(eps_0*m_e)) #Electron plasma frequency
    Omega_he=q_p*Bmag/(4*m_amu) #Helium cyclotron frequency
    Omega_ne=q_p*Bmag/(20*m_amu) #Neon cyclotron frequency
    Omega_e=q_e*Bmag/(m_e) #Electron cyclotron frequency
   
    #Calculate R,L and P
    R=1-((Pi_e**2/w**2)*(w/(w+Omega_e)))-((Pi_he**2/w**2)*(w/(w+Omega_he)))-((Pi_ne**2/w**2)*(w/(w+Omega_ne))) #Right-hand polarized wave
    L=1-((Pi_e**2/w**2)*(w/(w-Omega_e)))-((Pi_he**2/w**2)*(w/(w-Omega_he)))-((Pi_ne**2/w**2)*(w/(w-Omega_ne))) #Left-hand polarized wave
    P=1-(Pi_e**2/(w*(w+1j*nu_e)))-(Pi_he**2/w**2)-(Pi_ne**2/w**2) #Unmagnetized plasma

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

@variable.array(complex = True, shape=(3,3))
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
    #Construct the tensor elements
    u_xx=1*(PML(y,'y')*PML(z,'z')/PML(x,'x'))
    u_xy=0*PML(z,'z')
    u_xz=0*PML(y,'y')

    u_yx=0*PML(z,'z')
    u_yy=1*(PML(z,'z')*PML(x,'x')/PML(y,'y'))
    u_yz=0*PML(x,'x')

    u_zx=0*PML(y,'y')
    u_zy=0*PML(x,'x')
    u_zz=1*(PML(x,'x')*PML(y,'y')/PML(z,'z'))

    #Construct the matrix
    muR=np.array([[u_xx,u_xy,u_xz],[u_yx,u_yy,u_yz],[u_zx,u_zy,u_zz]])

    #MFEM uses column major for the matrix
    return muR