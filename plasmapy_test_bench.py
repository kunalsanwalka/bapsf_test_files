# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:45:18 2019

@author: kunalsanwalka

Test bench for the plasmapy library
"""

import plasmapy as pp
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

#Arbitrary B field
Bx=1
By=0
Bz=0

#Other plasma variables
species=['e','He+'] #Type of plasma
n=[1e12*u.cm**-3,1e12*u.cm**-3] #Number densities
f=np.logspace(start=4,stop=10,num=4001) #Frequency ranges
omega_RF=f*(2*np.pi)*(u.rad/u.s) #Angular Frequency

#Convert to spherical co-ordinates
Bnorm=np.sqrt(Bx**2+By**2+Bz**2)
theta=np.arccos(Bz/Bnorm)
phi=np.pi/2
if Bx!=0:
    phi=np.arctan(By/Bx)
    
#Find the SPD values
S,D,P=pp.physics.cold_plasma_permittivity_SDP(Bnorm,species,n,omega_RF[0])

#Construct the epsilon_r matrix in the local frame
epsRLoc=np.array([[S,1j*D,0],[-1j*D,S,0],[0,0,P]])

#Apply the rotation matrices
#Rotate away the phi
phiRotMat=np.array([[np.cos(phi),np.sin(phi),0],[-np.sin(phi),np.cos(phi),0],[0,0,1]])
temp=np.matmul(phiRotMat,epsRLoc)
epsRNoPhi=np.matmul(temp,np.linalg.inv(phiRotMat))
#Rotate away the theta
thetaRotMat=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
temp=np.matmul(thetaRotMat,epsRNoPhi)
epsRGlobal=np.matmul(temp,np.linalg.inv(thetaRotMat))

#Print the results
with np.printoptions(precision=3, suppress=True):
    print('B Field vector is-')
    print([Bx,By,Bz])
    print('eps_r in local co-ordinates')
    print(epsRLoc)
    print('eps_r in global co-ordinates')
    print(epsRGlobal)

####################
####################

##Define the experimental parameters
#B=0.1*u.T #Magnetic field strength
#species=['e','He+'] #Type of plasma
#n=[1e12*u.cm**-3,1e12*u.cm**-3] #Number densities
#f=np.logspace(start=4,stop=10,num=4001) #Frequency ranges
#omega_RF=f*(2*np.pi)*(u.rad/u.s) #Angular Frequency
#
##Calculate S,P,D values
#S,D,P=pp.physics.cold_plasma_permittivity_SDP(B,species,n,omega_RF)
#
##Calculate for a specific frequency
#omega_RF_1=1e4*(2*np.pi)*(u.rad/u.s)
#S_1,D_1,P_1=pp.physics.cold_plasma_permittivity_SDP(B,species,n,omega_RF_1)
#print('The S value is-')
#print(S_1)
#print('The D value is-')
#print(D_1)
#print('The P value is-')
#print(P_1)
#
##Construct sigma
#sigma=[[0,0,0],[0,0,0],[0,0,0]] #Initialize
#sCoEff=omega_RF_1*pp.constants.eps0
#sigma[0][0]=sCoEff*(1-S_1)*1j
#sigma[0][1]=sCoEff*(D_1)
#sigma[1][0]=sCoEff*(-D_1)
#sigma[1][1]=sCoEff*(1-S_1)*1j
#sigma[2][2]=sCoEff*(1-P_1)
##Print sigma
#print('The relative conductivity tensor is-')
#for x in sigma:
#    print (*x,sep="\t")
#
##Plot S,P,D values as a function of the frequency
##Filter positive and negative values for display purposes
#S_pos = S * (S > 0)
#D_pos = D * (D > 0)
#P_pos = P * (P > 0)
#S_neg = S * (S < 0)
#D_neg = D * (D < 0)
#P_neg = P * (P < 0)
##Get rid of 0 values
#S_pos[S_pos == 0] = np.NaN
#D_pos[D_pos == 0] = np.NaN
#P_pos[P_pos == 0] = np.NaN
#S_neg[S_neg == 0] = np.NaN
#D_neg[D_neg == 0] = np.NaN
#P_neg[P_neg == 0] = np.NaN
#
##Plot the actual data
#plt.figure(figsize=(12, 8))
#plt.semilogx(f, abs(S_pos),
#             f, abs(D_pos),
#             f, abs(P_pos), lw=2)
#plt.semilogx(f, abs(S_neg), '#1f77b4',
#             f, abs(D_neg), '#ff7f0e',
#             f, abs(P_neg), '#2ca02c', lw=2, ls='--')
#plt.yscale('log')
#plt.grid(True, which='major')
#plt.grid(True, which='minor')
##plt.ylim(1e-4, 1e8)
#plt.xlim(1e4, 1e10)
#plt.legend(('S > 0', 'D > 0', 'P > 0', 'S < 0', 'D < 0', 'P < 0'),
#           fontsize=16, ncol=2)
#plt.xlabel('RF Frequency [Hz]', size=16)
#plt.ylabel('Absolute value', size=16)
#plt.tick_params(labelsize=14)

####################
####################