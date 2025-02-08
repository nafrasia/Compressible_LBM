"""
====================================================================
 Plots velocity profile and computes viscosity for an incompressible
 Poiseuille flow

 Author: Navid Afrasiabian <nafrasia@uwo.ca>

 License: MIT, 2025
====================================================================

Parameters
-----------
savefig: str
    If savefig is provided on command line, 
    figures are saved to hard drive

"""
#--------------------------------------------------------------------
# Load necessary packages
#--------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sc
import os
import sys

#======================
# Function definition
#======================

def parabola(x,a,b,c):
    ''' Returns values of parabola'''
    return a*x**2+b*x+c;

def analytic_profile(z, a, rho, visc, H):
    ''' Returns the analytic velocity profile of a Poiseuille flow''' 
    return -(1/2/visc)*rho*a*z*(z-H)

#-----Command-line flag for saving figures-----
saveflag = 0

for flag in sys.argv:
    if flag == "savefig":
        saveflag = 1

#--------Define directory variables------
    cwd = os.getcwd()
    read_dir = os.path.join(cwd, 'data')
    write_dir = os.path.join(cwd, 'output')
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

# Lists of data files
tau1_files = ["Poiseuilleg1e-05T0.1tau1eq.csv","Poiseuilleg1e-05T0.2tau1eq.csv","Poiseuilleg1e-05T0.3tau1eq.csv","Poiseuilleg1e-05T0.4tau1eq.csv"]
tau2_files = ["Poiseuilleg1e-05T0.1tau2eq.csv","Poiseuilleg1e-05T0.2tau2eq.csv","Poiseuilleg1e-05T0.3tau2eq.csv","Poiseuilleg1e-05T0.4tau2eq.csv"]

# A list of data file lists is created to loop over and plot the viscosity versus temperature plots
filenamelist = [tau1_files, tau2_files]
taucounter = 0  # Counter to keep track of viscosity-temp plots

# List of relaxation times
taulist=[1,2]
ay = 0.00001    # Acceleration/driving force
dx=2            # Spatial resolution/ Grid spacing

# Plot Styling parameters (We need max of 4 but I defined 10 in case needed in the future)
colour = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
markers = ['o', 'D', 's', '*', 'p','P','d','>','<', '^']


for filenames in filenamelist:
    Tlist = np.array([])
    vislist = np.array([])
    i=0
    for file in filenames:
        # Read data
        bcdf = pd.read_csv(os.path.join(read_dir,file), header=0, names=["z","rho","temp","vx", "vy", "vz"])

        # Fit the data to a parabola to find the velocity profile
        init_guess=[1,0,0]
        coeff, pov = sc.curve_fit(parabola, np.array(bcdf['z']), np.array(bcdf["vy"]), p0=init_guess)

        # Find the exact location of the boundaries (i.e. where velocity is zero) by finding the root of the velocity profile
        bcpos = sc.fsolve(parabola,[0,160],args=(coeff[0], coeff[1], coeff[2]))

        # Interpolate and extraploate the data to get the full velocity profile
        z = np.linspace(bcpos[0], bcpos[1], 2000)
        vyfit = coeff[0]*z**2+coeff[1]*z+coeff[2]
        
        # Compute viscosity using Poiseuille relation (see section III.A of the paper)
        H=bcpos[1]-bcpos[0]
        umax = np.amax(vyfit)
        zmax = z[np.argmax(vyfit)]
        rho = np.mean(bcdf["rho"])
        viscosity = -rho*ay*(zmax)*(zmax-H)/2/umax
        vislist = np.append(vislist, viscosity)

        # Create a list of temperatures for plotting
        Tlist = np.append(Tlist, bcdf["temp"][0])
        
        plt.figure(taucounter,figsize=(8,6))
        # Plot fitted parabola
        plt.plot(z, vyfit, color = colour[i], linestyle='-',label='fit')
        # Plot the original data (placed after plotting the fit data to bring the data forward for better visualization)
        plt.plot(np.array(bcdf['z']), np.array(bcdf["vy"]), color = f'{colour[i]}', marker=markers[i], markevery=40, linestyle= 'None', label=f"T={Tlist[i]:0.2f}") #plotting original data
        # Plot the analytical solution (It is exactly the same as the fit data so it wasn't shown in the paper)
        plt.plot(z, analytic_profile(z-bcpos[0], ay, rho, viscosity, H), color = colour[i], linestyle='--',label='analytic')
        
        i+=1

    plt.xlabel(r"z $(\mu m)$", fontsize=15)
    plt.ylabel(r"$u_y (\mu m/\mu s)$", fontsize = 15)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if saveflag:
        plt.savefig(os.path.join(write_dir, f"u_poiseuille_tau{taulist[taucounter]}.png"), bbox_inches="tight", dpi=300)
    
    taucounter += 1

    # Fit the viscosity vs. temperature data to a line
    viscoeff = np.polyfit(Tlist, vislist, 1)
    
    # Plot viscosity as a funciton of temperature
    plt.figure(100, figsize=(8,6))
    plt.plot(Tlist, vislist, 'o', label=fr'$\tau = ${taucounter}')

    # Plot the fitted line
    plt.plot(np.linspace(Tlist[0], Tlist[-1], 100), np.linspace(Tlist[0], Tlist[-1], 100)*viscoeff[0]+viscoeff[1], label=f'{viscoeff[0]:0.2f} T + {viscoeff[1]:0.2f}')
    plt.xlabel(r"T $(m^2/s^2)$", fontsize=15)
    plt.ylabel(r"$\eta (\frac{pg}{\mu m \cdot \mu s})$", fontsize = 15)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if saveflag:
        plt.savefig(os.path.join(write_dir, f"viscosity_poiseuille.png"), bbox_inches="tight", dpi=300)
if not saveflag:
    plt.show()
