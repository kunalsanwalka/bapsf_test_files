# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:08:53 2020

@author: kunalsanwalka
"""

import numpy as np
import matplotlib.pyplot as plt

savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/PML_stretch.png'

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

        #Set the magnitude and order of the co-ordinate stretch
        magS=5
        ordS=1

        #Initialize S_r
        S_r=1   
        
        if r>0:
            #Postive Region PML

            #Take the absolute value of r
            rAbs=np.abs(r)

            #Define L_r and L_PML (define where the PML begins and its depth)
            PMLbegin=8#m (LAPD is 6m long)
            PMLdepth=2#m (PML Layer is 2m long)

            #Define S_r' and S_r'' (the real and complex stretch respectively)
            realStretch=(magS*(rAbs-PMLbegin))**ordS #Damps evanescent waves
            complexStretch=(magS*(rAbs-PMLbegin))**ordS #Damps real waves

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
            PMLbegin=8#m (LAPD is 6m long)
            PMLdepth=2#m (PML Layer is 2m long)

            #Define S_r' and S_r'' (the real and complex stretch respectively)
            realStretch=(magS*(rAbs-PMLbegin))**ordS #Damps evanescent waves
            complexStretch=(magS*(rAbs-PMLbegin))**ordS #Damps real waves

            #Define p_r (ramping factor)
            rampFact=2

            #Calculate S_r
            if rAbs>PMLbegin:
                S_r=1+(realStretch+1j*complexStretch)*(((rAbs-PMLbegin)/PMLdepth)**rampFact)
        return S_r
    else:
        return 1
    
def PML2(r,co_ord):    
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
            realStretch=11 #Damps evanescent waves
            complexStretch=10 #Damps real waves

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
            realStretch=5 #Damps evanescent waves
            complexStretch=5 #Damps real waves

            #Define p_r (ramping factor)
            rampFact=2

            #Calculate S_r
            if rAbs>PMLbegin:
                S_r=1+(realStretch+1j*complexStretch)*(((rAbs-PMLbegin)/PMLdepth)**rampFact)
        return S_r
    else:
        return 1
    
#z values
zArr=np.linspace(-10,10,1001)
#PML stretch values
PMLArr=[]

for z in zArr:
    PMLArr.append(PML(z,'z'))
#Convert to numpy
PMLArr=np.array(PMLArr)

# =============================================================================
# Plotting
# =============================================================================

#Create the subplots
fig,(ax1,ax2)=plt.subplots(1,2,sharey=True)
fig.subplots_adjust(wspace=0.05)

#Plot the data
#Main data
ax1.plot(zArr,np.real(PMLArr))
ax1.plot(zArr,np.imag(PMLArr))
#PML Ramps
ax2.plot(zArr,np.real(PMLArr),label='Real')
ax2.plot(zArr,np.imag(PMLArr),label='Imaginary')
#PML Boundary
ax2.plot([8,8],[-0.5,12],label='PML Boundary',linestyle='dashed',color='k')

#Set axes limits
ax1.set_xlim(0,3.2)
ax2.set_xlim(6.8,10)
ax1.set_ylim(-0.5,12)

#Hide the spines common between plots
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax1.tick_params(labelleft=True)
ax2.yaxis.tick_right()

#Add diagnoal slashes
d=0.015
kwargs=dict(transform=ax1.transAxes,color='k',clip_on=False)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)        # top-right diagonal
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (-d, +d), **kwargs)              # top-left diagonal
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)        # bottom-left diagonal

#Add grid
ax1.grid(True)
ax2.grid(True)

#Add axes labels
fig.text(0.5, 0.02, 'Z [m]', ha='center', va='center')
ax1.set_ylabel('Stretch Value [arb. u.]')

#Add legend
ax2.legend(bbox_to_anchor=(1.91,1.03),loc='upper right')

plt.savefig(savepath,dpi=600,bbox_inches='tight')
plt.show()