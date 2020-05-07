# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:22:38 2019

@author: kunalsanwalka

This program converts the dielectric tensor from the local frame to the global cartesian frame

##########
Verified
##########

"""

import plasmapy as pp
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

#Plasma variables
species=['e','He+'] #Type of plasma
n=[1e12*u.cm**-3,1e12*u.cm**-3] #Number densities
f=np.logspace(start=4,stop=10,num=4001) #Frequency ranges
omega_RF=f*(2*np.pi)*(u.rad/u.s) #Angular Frequency

#Arbitrary B field
Bx=1
By=0
Bz=0

#Convert cartesian to spherical
Bnorm=np.sqrt(Bx**2+By**2+Bz**2)
theta=np.arccos(Bz/Bnorm)
phi=np.pi/2
if Bx!=0:
    phi=np.arctan(By/Bx)
    
#Find the SPD values in the local frame
S,D,P=pp.physics.cold_plasma_permittivity_SDP(Bnorm,species,n,omega_RF[0])
#Construct the epsilon_r matrix in the local frame
epsRLoc=np.array([[S,1j*D,0],[-1j*D,S,0],[0,0,P]])

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

#Print the results
with np.printoptions(precision=3, suppress=True):
    print('B Field vector is-')
    print([Bx,By,Bz])
    print('phi= '+str(phi))
    print('theta= '+str(theta))
    print('eps_r in local co-ordinates')
    print(epsRLoc)
    print('eps_r in global co-ordinates')
    print(epsR)