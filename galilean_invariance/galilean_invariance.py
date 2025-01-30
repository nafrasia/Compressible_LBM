#====================================================================
# This code plots HMLBM, standard, and analytical solutions of
# the Couette flow under gravitational field
# 
# Author: Navid Afrasiabian <nafrasia@uwo.ca>
#
# License: MIT 2025
#====================================================================

#-----------------------------------------------
# Importing necessary packages
#-----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.optimize as sc
import sys

# Get the current directory
cwd= os.getcwd()

#------------------------------------------------
# Define functions
#------------------------------------------------
def u_profile(z, ut, ub, g, T, H):
    ''' Analytical velocity profile (see section III.C of the paper)'''
    return (ut-ub)*(1-np.exp(-(g/T)*(z)))/(1-np.exp(-(g/T)*(H)))+ub

def scaled_u(z, g, T, H):
    return (1-np.exp(-(g/T)*(z)))/(1-np.exp(-(g/T)*(H)))

def shifted_scaled_u(z, g, T, H, d):
    return (1-np.exp(-(g/T)*(z+d)))/(1-np.exp(-(g/T)*(H+d)))

def plot_sim(file, x, y, headers = ["z","rho","temp","vx", "vy", "vz"], scaled=False, line_style='-', color = 'tab:blue', legend=None, ax=None):
    # Read the data over line file
    if len(headers) != 0:
        data_df = pd.read_csv(file, header=0, names=headers)
    else:
        data_df = pd.read_csv(file)
        headers = data_df.columns
    
    if x not in headers:
        raise ValueError(f"Variable {x} not found")
    if y not in headers:
        raise ValueError(f"Variable {y} not found")

    if ax is None:
        fig, ax = ax.subplots()
    # Plot the data
    if scaled:# Normalize the velocity
        ax.plot(data_df[x], (np.array(data_df[y])-ub[i])/(ut[i]-ub[i]), linestyle=line_style, color = color, label=legend)

    else:# Plot the original data with no scaling
        ax.plot(data_df[x], np.array(data_df[y]), linestyle=line_style, color = color, label=legend)

    return data_df

def plot_analytic(loc_bot, loc_top, velo_bot, velo_top, g, T, resolution = 1000, scaled=False, line_style='none', color = 'k', legends='analytical solution', ax=None):
    
    ''' Computes and plots the analytical solution to Couette flow in a gravitation field (see section III.C)'''

    if ax is None:
        fig, ax = plt.subplots()
        
    if scaled:
        analytic = scaled_u(np.linspace(0,loc_top-loc_bot,resolution), g, T, loc_top-loc_bot)
        ax.plot(np.linspace(0, 1,resolution), analytic,
                linestyle = 'none', marker='*', markevery=50, color = color, label="analytical solution")
    else:
        analytic = u_profile(np.linspace(0,loc_top-loc_bot,resolution), velo_top, velo_bot, g, T, (loc_top-loc_bot))
        ax.plot(np.linspace(loc_bot,loc_top, resolution), analytic,
                    linestyle = 'none', marker='*', markevery=35, color = color, label="analytical solution")
    
    return analytic

#---------------------------------------
# Main code
#---------------------------------------
if __name__ == '__main__':

    # This command-line flag allows user to save the output figure     
    if len(sys.argv) > 2:
        print("too many arguments! Exit")
        exit();
    elif len(sys.argv) < 2:
        savefig = 0
    else:
        if sys.argv[1] == 'savefig':
            savefig = 1
    # hmlbm stands for Higher Moment Lattice Boltzmann Method
    hmlbmFiles=["Couette_HMLBM_U0.5.csv","Couette_HMLBM_U1.csv"]
    standardFiles=["Couette_standard_U0.5.csv", "Couette_standard_U1.csv"]
    
    #----Graph styling----
    style_dict = {"hmlbm":{"line":"-", "color":"tab:blue", "label":"HMLBM"}, "standardlb":{"line":"--", "color":"tab:orange", "label":"standard LBM"}, "analytic":{"line":"none", "color":"k", "label":"analytical solution"}}
    
    #-------Set physical parameters---------    
    g = -980 # gravitational acceleration (in z direction)
    T = 1666.667 # Temperature in energy units
    
    scaling = False

    myfig, myax = plt.subplots()
    #----Loop to iterate over simulations and plot them-----
    for hmlbmFile, standardFile in zip(hmlbmFiles, standardFiles):
        #-------Plot velocity profile for standard model (Using LATBOLTZ package of lammps)-----
        plot_sim(os.path.join(cwd,os.path.join("standardLB",standardFile)),x="z", y="vy", headers=["z", "rho", "vx", "vy", "vz"], scaled = False,
                 line_style=style_dict["standardlb"]["line"], color=style_dict["standardlb"]["color"], legend=style_dict["standardlb"]["label"], ax=myax)
    
        #------Plot HMLBM velocity profile-----
        data = plot_sim(os.path.join(cwd,os.path.join("HMLBM",hmlbmFile)), x = "z", y= "vy", headers=["z", "rho", "temp", "vx", "vy", "vz"], scaled = False,
                 line_style = style_dict["hmlbm"]["line"], color = style_dict["hmlbm"]["color"], legend = style_dict["hmlbm"]["label"], ax = myax)
    
        #------Plot analytical solutions-------
        z_b, z_t, v_b, v_t = np.min(data["z"]), np.max(data["z"]), np.min(data["vy"]), np.max(data["vy"])
        analytic = plot_analytic(z_b, z_t, v_b, v_t, g, T, resolution=len(data["z"]), scaled=False, ax=myax)
    
        print(f"RMSE = {np.sqrt(np.mean((analytic-data["vy"])**2))}")
    
    myax.legend()
    if scaling==True:
        myax.set_ylabel(r"$(u - u_{bot})/(u_{top} - u_{bot})$")
        myax.set_xlabel(r"$z(\mu m)$")
    else:
        myax.set_ylabel(r"$u_y (cm/s)$")
        myax.set_xlabel(r"$z(cm)$")
    
    if savefig:
        plt.savefig(os.path.join(cwd, "Galilean_Invariance.png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()
