import numpy as np
import matplotlib.pyplot as plt

savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/Bz_Background_Axial_Amp.png'

zMax=7 #Size of the plasma chamber

def Bfield(x,y,z):
    import numpy as np
    
    #Base Case
    B=[0,0,0.1]
    
    #Convert z to the absolute value as the magnetic field is symmetric
    z=np.abs(z)
    
    B[2]*=(-np.tanh(2.5*(z-zMax))+1)/2
    
    return B

#Plot the field lines as a function of z
zRange=np.arange(-10,10.01,0.01)
bFieldArr=[]
for zVal in zRange:
    bFieldArr.append(Bfield(0,0,zVal)[2]) #We just need the z-component

plt.plot(zRange,bFieldArr)
plt.xlabel('Z [m]')
plt.ylabel('$B_z$')
plt.title('Magnetic field as a function of the axial position')
plt.grid(True)
plt.savefig(savepath,dpi=300,bbox_to_anchor='tight')
plt.show()

print(Bfield(0,0,10))