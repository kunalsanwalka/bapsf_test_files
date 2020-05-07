# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:56:00 2019

@author: kunalsanwalka

This program opens HDF5 files from the Petra-M results and plots them
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def dataArr(filename):
    """
    This function takes the name of an hdf5 file and returns the relevant data arrays
    
    Data in the hdf5 file is organized as-

    Group 1- page1
        There is no data in this group, we can ignore it
        
    Group 2- page1.axes1.solid1
        This group has the following subgroups-
        Group 1- cdata
            Contains the data for the slice
            cdata has 2 subsequent subgroups-
                Group 1- Imag
                    Stores the complex part of the number
                Group 2- Real
                    Stores the real part of the number
        Group 2- idxset
            Do not know what data is stored here
        Group 3- vertices
            Contains the co-ordinates for the slice
    
    Args:
        filename: Name of the file (str)
    Returns:
        cdata: Complex valued solution of the problem (np.array)
        xVals: x-coordinate of the data points (np.array)
        yVals: y-coordinate of the data points (np.array)
    """
    #Open the file
    f=h5py.File(filename,'r')
    
    #Initialize the data arrays
    cdata=[]
    idxset=[]
    vertices=[]
    
    #Open groups in the file
    for group in f.keys():
#        print('Group- '+group)
        
        #Get the group
        currGroup=f[group]
        
        #Open keys in the group
        for key in currGroup.keys():
#            print('Key- '+key)
    
            #Append the data to the respective arrays
            if key=='cdata(Complex)':
                cdataGroup=currGroup[key]
                
                imag=[]
                real=[]
                #Open the keys in cdata
                for subkey in cdataGroup.keys():
#                    print('Subkey- '+subkey)
                    
                    #Get the real and imaginary parts of the array
                    if subkey=='Imag':
                        imag=cdataGroup[subkey][()]
                    elif subkey=='Real':
                        real=cdataGroup[subkey][()]
                
                #Convert lists to numpy arrays
                imag=np.array(imag)
                real=np.array(real)
                #Get the cdata value
                cdata=real+1j*imag
                
            elif key=='idxset':
                idxset=currGroup[key][()]
            elif key=='vertices':
                vertices=currGroup[key][()]
    
    #Remove the unwanted component from the vertices
    #Vertices are stored in a 3 element array containing (x,y,z) but we only
    #need 2 to make flat plots, therefore we need to remove one based on the
    #plane on which the data is being plotted
    xVals=[]
    yVals=[]
    newVertices=[]
    for vertex in vertices:
        xVals.append(vertex[0])
        yVals.append(vertex[2])
        newVertices.append([vertex[0],vertex[1]])
    vertices=newVertices
    
    #Convert to numpy arrays
    cdata=np.array(cdata)
    xVals=np.array(xVals)
    yVals=np.array(yVals)
    
    #Close the file
    f.close()
    
    return cdata, xVals, yVals

def animPlot(cdata,xvals,yvals,savepath,plotTitle,omega):
    """
    This functions takes the data of a component and plots the respective animation
    
    Args:
        cdata: The data to be plotted (np.array)
        xvals, yvals: Arrays with the locations of the data points (np.array)
        savepath: Destination address of the animation
        plotTitle: Title of the plot
        omega: Angular antenna frequency
    Returns:
        Saves the animation as a .html in the location specifed
    """
    #Animate the plot
    #Setup the plot
    fig,ax=plt.subplots(figsize=(8,12))
#    plt.axes(xlim=(-1,1),ylim=(-1,1))
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.title(plotTitle)
    
    #Create the time array
    tArr=np.linspace(0,2*np.pi/omega,100)
    #Create exponent array
    expArr=np.e**(-1j*omega*tArr)
    
    #Sanity check to make sure the points are well spaced
#    print(cdata)
#    print(expArr)
#    
#    X = [x.real for x in expArr]
#    Y = [x.imag for x in expArr]
#    plt.scatter(X,Y, color='red')
#    plt.show()
#    
#    currData=cdata*expArr[0]
#    print(np.absolute(currData))
#    print('********************************************')

    #Create the animation function
    def animate(i):
        #Create the data to be plotted
        currData=cdata*expArr[i]
        #Get the real part of the complex array
        realData=np.real(currData)
        
#        print('Frame Number- '+str(i))
        
        #Plot the data
        cont=plt.tricontourf(xvals,yvals,realData,10)
        
        #Add the colorbar
        if i==0:
            plt.colorbar()
        
        return cont
    
    anim=animation.FuncAnimation(fig,animate,frames=(len(expArr)-1),blit=False,repeat=False)
    anim.save(savepath)
    
freq=250*1000
omega=2*np.pi*freq
filename='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Data/By_XZ_Plane_O4.hdf'
savepath='C:/Users/kunalsanwalka/Documents/UCLA/BAPSF/Plots_and_Animations/By_XZ_Plane.html'
plotTitle='$B_y$; freq=250KHz; XZ_Plane'

#Get the data
cdata, xVals, yVals=dataArr(filename)

#Animate the solution
animPlot(cdata,xVals,yVals,savepath,plotTitle,omega)