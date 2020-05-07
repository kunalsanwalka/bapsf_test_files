# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

#Experimental Values
magB=0.1 #T
numDens=1e18 #m^{-3}
kPerpDelE=0.6*1e-4

def alfvenPhaseVel(magB,numDens):
    """
    This function calculates the alfven wave phase velocity

    Args:
        magB: Strength of the background magnetic field
        numDens: Density of the plasma
    Returns:
        The Alfven wave phase velocity
    """

    return magB/np.sqrt(const.mu_0*numDens*const.physical_constants['alpha particle mass'][0])

def ionCycFreq(magB):
    """
    This function calcualtes the ion cyclotron frequency

    Args:
        magB: Strength of the background magnetic field
    Returns:
        The ion cyclotron frequency
    """

    return const.e*magB/(const.physical_constants['alpha particle mass'][0])

def ionPlasmaFreq(density):
    """
    This function calculates the ion plasma frequency

    Args:
        density: The ion density in m^{-3}
    Returns:
        The ion cyclotron frequency
    """

    #Numerator
    num=density*const.e**2
    #Denominator
    den=const.epsilon_0*const.physical_constants['alpha particle mass'][0]

    return np.sqrt(num/den)

def ionSkinDepth(density):
    """
    This function calcuates the ion skin depth

    Args:
        density: Density of the plasma in m^{-3}
    Returns:
        The ion skin depth
    """

    return const.c/ionPlasmaFreq(density)

#k_par dispersion relation

#2 Fluid Dispersion Relation Model
def twoFluidDisp(omega,magB,numDens,kPerpDelE):
    """
    This function calculated the k-par dispersion relation based on the 2 Fluid Model
    Note: All units are SI

    Args:
        omega: Angular frequency of the antenna
        magB: Strength of the background magnetic field
        numDens: Density of the plasma
        kPerpDelE: Normalized perpendicular wavenumber
    Returns:
        The parallel wavenumber
    """

    #Ion cyclotron frequency
    Omega_ci=ionCycFreq(magB)

    #Alfven Wave Velocity
    va=alfvenPhaseVel(magB,numDens)

    #1st Term
    t1=omega*Omega_ci/va

    #2nd Term
    t2=(1+kPerpDelE**2)/(Omega_ci**2-omega**2) #0.025 k_perp*delta_e value from the raw data

    return t1*np.sqrt(t2)

#MHD Dispersion Relation Model
def MHDDisp(omega,magB,numDens):
    """
    This function calculated the k-par dispersion relation based on the MHD Model
    Note: All units are SI

    Args:
        omega: Angular frequency of the antenna
        magB: Strength of the background magnetic field
        numDens: Density of the plasma
    Returns:
        The parallel wavenumber
    """

    #Alfven Wave Velocity
    va=alfvenPhaseVel(magB,numDens)

    return omega/va

#Initalize Arrays
#Simulation Arrays
freq=np.array([50,100,150,200,250,275,300,350])*1000 #KHz
omega_sim=freq*2*np.pi

#Read the values of k_|| from the various savefiles
mesh_100=[]
mesh_75=[]
mesh_60=[]
with open('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/k_par_mesh_100.txt','r') as f:
    for i in range(len(omega_sim)):
        mesh_100.append(float(f.readline()))
with open('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/k_par_mesh_75.txt','r') as f:
    for i in range(len(omega_sim)):
        mesh_75.append(float(f.readline()))
with open('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/k_par_mesh_60.txt','r') as f:
    for i in range(len(omega_sim)):
        mesh_60.append(float(f.readline()))

#Theory Arrays
omega_theory=np.linspace(min(omega_sim),max(omega_sim),100)
MHDArr=[]
twoFluidArr=[]

#Calculate the values predicted by theory
for omega in omega_theory:
    twoFluidArr.append(twoFluidDisp(omega,magB,numDens,kPerpDelE))
    MHDArr.append(MHDDisp(omega,magB,numDens))

#Normalize the arrays
#Normalize the angular frequency
omega_sim/=ionCycFreq(magB)
omega_theory/=ionCycFreq(magB)
#Normalize the angular wavenumber
mesh_100=np.array(mesh_100)*ionSkinDepth(numDens)
mesh_75=np.array(mesh_75)*ionSkinDepth(numDens)
mesh_60=np.array(mesh_60)*ionSkinDepth(numDens)
MHDArr=np.array(MHDArr)*ionSkinDepth(numDens)
twoFluidArr=np.array(twoFluidArr)*ionSkinDepth(numDens)

#Plot the results
p1=plt.plot(mesh_100,omega_sim,label='Mesh=10cm')
p1=plt.plot(mesh_75,omega_sim,label='Mesh=7.5cm')
p1=plt.plot(mesh_60,omega_sim,label='Mesh=6cm')
p1=plt.scatter(mesh_100,omega_sim)
p1=plt.scatter(mesh_75,omega_sim)
p1=plt.scatter(mesh_60,omega_sim)
#p1=plt.plot(MHDArr,omega_theory,label='MHD Theory')
p1=plt.plot(twoFluidArr,omega_theory,label='2 Fluid Theory')
p1=plt.grid(True)
p1=plt.title(r'$k_{||}$ Dispersion Relation',fontsize=18)
p1=plt.xlabel(r'Normalized Angular Wavenumber $[k_{||}\delta_{i}]$',fontsize=14)
p1=plt.ylabel(r'Normalized Angular Frequency $[\omega/\omega_c]$',fontsize=14)
#p1=plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.ylim((0,1))
plt.xlim((0,2.5))
p1=plt.legend(loc='best')
#p1=plt.savefig('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/k_par_dispersion_relation.png',dpi=600)
p1=plt.show()

##k_perp dispersion relation

##2 Fluid Dispersion Relation Model
#def kPerp2Fluid(omega,magB,numDens,kPar):
#    """
#    This function calculated the k-perp dispersion relation based on the 2-fluid model
#    Note: All units are SI

#    Args:
#        omega: Angular frequency of the antenna
#        magB: Strength of the background magnetic field
#        numDens: Density of the plasma
#        kPar: Axial angular wavenumber
#    Returns:
#        The normalized perpendicular wavenumber
#    """

#    #Alfven wave phase velocity
#    va=alfvenPhaseVel(magB,numDens)

#    #Ion Cyclotron frequency
#    Omega_ci=ionCycFreq(magB)

#    #1st term
#    t1=(kPar*va/omega)**2
#    #2nd term
#    t2=1-(omega/Omega_ci)**2

#    print(t1)
#    print(t2)
#    print('************')

#    return np.sqrt(t1*t2-1)

#kPar=4.87
#numDensArr=np.linspace(1e18,5e18,500)
#omega=2*np.pi*350*1000

#print(kPerp2Fluid(omega,magB,1e18,1.5))

#kPerpArr=[]
#vaArr=[]
#for dens in numDensArr:
#    kPerpArr.append(kPerp2Fluid(omega,magB,dens,kPar))
#    vaArr.append(alfvenPhaseVel(magB,numDens))

#plt.plot(kPerpArr,vaArr)
#plt.xlabel(r'$k_{\perp}\delta_e$')
#plt.ylabel(r'v_A')
#plt.grid(True)
#plt.show()