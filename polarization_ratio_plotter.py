# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:16:20 2020

@author: kunalsanwalka
"""

import numpy as np
import matplotlib.pyplot as plt

#Frequency of the simulation
freq=480 #KHz

#Collisionalities of the simulations
nu_eArr=[0.5e6,2e6,5e6] #Hz

#Savepath
savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/LH_fraction_XY_integral_r_50cm.png'

#Create the plot
plt.figure(figsize=(15,4))

#Add each line
for nu_e in nu_eArr:
    
    #Open the arrays
    ratioArr=np.loadtxt('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/freq_'+str(freq)+'KHz_col_'+str(int(nu_e/1000))+'KHz/ratioArr_r_50cm.txt')
    zPosArr=np.loadtxt('C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/freq_'+str(freq)+'KHz_col_'+str(int(nu_e/1000))+'KHz/zPosArr.txt')
    
    #Plot the LH fraction
    plt.plot(zPosArr,ratioArr,label=r'$\nu_{ei}$='+str(int(nu_e/1000))+'KHz')
    
#Add the axes labels
plt.xlabel('Z [cm]')
plt.ylabel(r'LH Fraction [$|B_L|^2/|B_{Tot}|^2$]',rotation=90)
plt.title('Effect of collisionality on LH fraction evolution')

#Axes parameters
plt.xlim(0,1000)
plt.ylim(0.4,0.6)
plt.xticks(np.arange(0,1001,100))

#Misc
plt.legend()
plt.grid(True)
plt.savefig(savepath,dpi=600,bbox_to_inches='tight')
plt.show()
plt.close()