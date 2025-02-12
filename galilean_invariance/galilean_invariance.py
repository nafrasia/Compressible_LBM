"""
====================================================================
This code plots HMLBM, standard, and analytical solutions of
the Couette flow under gravitational field
Author: Navid Afrasiabian <nafrasia@uwo.ca>

License: MIT 2025
====================================================================

Parameters
----------
savefig: str
    If savefig is passed, the figures are saved to hard drive

"""
#-----------------------------------------------
# Importing necessary packages
#-----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.optimize as sc
import sys


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

def rho_profile(z, rho_init, g, T, H):
    return (rho_init*H*g)/(T*(np.exp(g*H/T)-1))*np.exp(g*z/T)

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

def plot_rho_analytic(rho_init, loc_bot, loc_top, g, T, resolution=1000, line_style='none', color='k', legends='analytical solution', ax = None):

    if ax is None:
        fig, ax = plt.subplots()

    analytic = rho_profile(np.linspace(0,loc_top-loc_bot,resolution), rho_init, g, T, loc_top-loc_bot)
    ax.plot(np.linspace(loc_bot,loc_top, resolution), analytic,
                    linestyle = 'none', marker='*', markevery=35, color = color, label="analytical solution")

#---------------------------------------
# Main code
#---------------------------------------
if __name__ == '__main__':
    savefig = 0
    vflag = 0
    rhoflag = 0

    # This command-line flag allows user to save the output figure     
    if len(sys.argv) > 4:
        print("too many arguments! Exit")
        exit();
    else:
        for flag in sys.argv:
            if flag == "savefig":
                savefig = 1
            
            if flag == "velocity":
                vflag = 1

            if flag == "density":
                rhoflag = 1
    #--------Define directory variables------
    cwd = os.getcwd()
    read_dir = os.path.join(cwd, 'data')
    write_dir = os.path.join(cwd, 'output')
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    
    # hmlbm stands for Higher Moment Lattice Boltzmann Method
    hmlbmFiles=["Couette_HMLBM_U0.5.csv","Couette_HMLBM_U1.csv"]
    standardFiles=["Couette_standard_U0.5.csv", "Couette_standard_U1.csv"]
    
    #----Graph styling----
    style_dict = {"hmlbm":{"line":"-", "color":"tab:blue", "label":"HMLBM"}, "standardlb":{"line":"--", "color":"tab:orange", "label":"standard LBM"}, "analytic":{"line":"none", "color":"k", "label":"analytical solution"}}
    
    #-------Set physical parameters---------    
    g = -980 # gravitational acceleration (in z direction)
    T = 1666.667 # Temperature in energy units
    rho_init = 0.001184 

    scaling = False

    vfig, vax = plt.subplots()
    rhofig, rhoax = plt.subplots()
    #----Loop to iterate over simulations and plot them-----
    if vflag == 1:
        for hmlbmFile, standardFile in zip(hmlbmFiles, standardFiles):
            #-------Plot velocity profile for standard model (Using LATBOLTZ package of lammps)-----
            plot_sim(os.path.join(read_dir,standardFile),x="z", y="vy", headers=["z", "rho", "vx", "vy", "vz"], scaled = False,
                     line_style=style_dict["standardlb"]["line"], color=style_dict["standardlb"]["color"], legend=style_dict["standardlb"]["label"], ax=vax)
        
            #------Plot HMLBM velocity profile-----
            data = plot_sim(os.path.join(read_dir, hmlbmFile), x = "z", y= "vy", headers=["z", "rho", "temp", "vx", "vy", "vz"], scaled = False,
                     line_style = style_dict["hmlbm"]["line"], color = style_dict["hmlbm"]["color"], legend = style_dict["hmlbm"]["label"], ax = vax)
            
            #------Plot analytical solutions-------
            z_b, z_t, v_b, v_t = np.min(data["z"]), np.max(data["z"]), np.min(data["vy"]), np.max(data["vy"])
            analytic = plot_analytic(z_b, z_t, v_b, v_t, g, T, resolution=len(data["z"]), scaled=False, ax=vax)
        
            print(f"RMSE = {np.sqrt(np.mean((analytic-data["vy"])**2))}")
            
        
        vax.legend()
        if scaling==True:
            vax.set_ylabel(r"$(u - u_{bot})/(u_{top} - u_{bot})$")
            vax.set_xlabel(r"$z(\mu m)$")
        else:
            vax.set_ylabel(r"$u_y (cm/s)$")
            vax.set_xlabel(r"$z(cm)$")
        
        if savefig:
            vfig.savefig(os.path.join(write_dir, "Galilean_Invariance.png"), dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    if rhoflag == 1:
        for hmlbmFile, standardFile in zip(hmlbmFiles, standardFiles):
            #------Plot HMLBM velocity profile-----
            data = plot_sim(os.path.join(read_dir, hmlbmFile), x = "z", y= "rho", headers=["z", "rho", "temp", "vx", "vy", "vz"], scaled = False,
                     line_style = style_dict["hmlbm"]["line"], color = style_dict["hmlbm"]["color"], legend = style_dict["hmlbm"]["label"], ax = rhoax)
            
            #-------Plot velocity profile for standard model (Using LATBOLTZ package of lammps)-----
            plot_sim(os.path.join(read_dir,standardFile),x="z", y="rho", headers=["z", "rho", "vx", "vy", "vz"], scaled = False,
                     line_style=style_dict["standardlb"]["line"], color=style_dict["standardlb"]["color"], legend=style_dict["standardlb"]["label"], ax=rhoax)
        
            z_b, z_t = np.min(data["z"]), np.max(data["z"])
            plot_rho_analytic(np.mean(data["rho"]), z_b, z_t, g, T, ax = rhoax) 

        rhoax.legend()
        rhoax.set_ylabel(r"$\rho (g/cm^3)$")
        rhoax.set_xlabel(r"$z(cm)$")

        if savefig:
            rhofig.savefig(os.path.join(write_dir, "Galilean_Invariance_rho.png"), dpi=300, bbox_inches="tight")
        else:
            plt.show()
