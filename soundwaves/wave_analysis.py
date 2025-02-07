"""
===================================================================
This code fits the decaying sound wave profiles and plots/computes
wave properties and fluid properties.
 
Author: Navid Afrasiabian <nafrasia@uwo.ca>

License: MIT 2025
===================================================================

Parameters/Arguments
-------------------- 
"density": density wave

"velocity": velocity wave

"temperature": temperature wave

variable: str
    Could be "eta" (viscosity), "k" (wave number), "T" (temperature)

"savefig": saves the plots onto the hard drive (in output folder)

"plot_raw_data": Creates a plot of the raw data

"plot_fit": Creates a plot of fitted functions

"""
import numpy as np
import pandas as pd
import scipy.optimize as sp
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import sys


#-----------------------------------------------
# Fitting functions. Most of these are fed into
# Scipy's curve_fit() to fit the data
#-----------------------------------------------
def fit_sine(t, a, b, w, p, c):
    ''' Decaying sine function'''
    return a*np.exp(-b*t)*np.sin(w*t+p)+c

def fit_cosine(t, a, b, w, p, c):
    ''' Decaying cosine function'''
    return a*np.exp(-b*t)*np.cos(w*t+p)+c

def fit_quad(k, a):
    ''' Quadratic equation with no first and zero power terms'''
    return a*k**2

def fit_exp(t, a, b, c):
    ''' Exponential decay equation'''
    return a*np.exp(-b*t)+c

def find_peak(x, y):
    """
        This function uses first and second order derivatives
        to find peaks/maxima of a curve. The derivatives are
        computed using finite difference method.

        Parameters
        -----------
        x: np.ndarray
            x-cooridnate of the data
        y: np.ndarray
            y-coordinate of the data
        
        Return
        ----------
        coordinates of the peaks
    """
    xpeak = np.array([])
    ypeak = np.array([])

    for i in range(1, len(x)-1):
        df1 = (y[i+1]-y[i])/(x[i+1]-x[i])
        db1 = (y[i]-y[i-1])/(x[i]-x[i-1])
        d2 = (y[i+1]-2*y[i]+y[i-1])/(2*(x[i]-x[i-1])**2)

        #print(i)
        if d2 < 0:
            if db1*df1 < 0:
                ypeak = np.append(ypeak, y[i])
                xpeak = np.append(xpeak, x[i])
    
    return (xpeak, ypeak)

def fit_wave(x, y, wave_type="sine", wave_init = np.zeros(5), fit_peak = True, peak_init = None, return_peak = False):
    """
    Fits the data with a decaying sinusoidal function

    Parameters
    ----------

    Return
    ----------
    """
    if wave_type == "cosine":
        param, pcov = sp.curve_fit(fit_cosine, x, y, p0=wave_init)
    elif wave_type == "sine":
        param, pcov = sp.curve_fit(fit_sine, x, y, p0=wave_init)
    else:
        raise ValueError(f"{wave_type} is an invalid wave_type")

    if fit_peak:
        if peak_init is None:
            raise ValueError("peak_init is required when fit_peak == True")
        else:
            tpeak, ypeak = find_peak(x,y)
            fit_rng = [4, 18] # Avoid fitting to potential transient region at the beginning or diminished peaks at the end
            expParam, expCov = sp.curve_fit(fit_exp, tpeak[fit_rng[0]:fit_rng[1]], ypeak[fit_rng[0]:fit_rng[1]], p0=peak_init)
            if return_peak:
                return ([param[0], expParam[1], param[2], param[3], param[4]], [tpeak, ypeak])
            else:
                return param[0], expParam[1], param[2], param[3], param[4]
    else:
        if return_peak:
            raise ValueError("Cannot return peak coordinates when fit_peak == False")

        return param

def compute_viscosity(gamma, k, rho):
    return gamma*rho/k/k

if __name__ == "__main__":
    #-------Global Variables-----------
    FIG_SIZE = (8,6)
    
    #-------Plotting flags-------------
    saveflag = 0
    showfit = 0
    showdata = 0
    simflag = 0
    
    #--------Command line flags--------
    if len(sys.argv) > 7:
        print("too many arguments! Exit")
        exit();
    else:
        i = 1 # 0 is for the program name.
        while (i < len(sys.argv)):
            iarg = sys.argv[i]
            if iarg == 'density':
                yflag = 1
                print("density analysis")
                i += 1
            elif iarg == 'velocity':
                yflag = 2
                print("velocity analysis")
                i += 1
            elif iarg == 'temperature':
                yflag = 3
                print("temperature analysis")
                i += 1
            elif iarg == 'savefig':
                saveflag = 1
                i += 1
            elif iarg == 'variable':
                if sys.argv[i+1] == 'eta':
                    simflag = 1
                elif sys.argv[i+1] == 'k':
                    simflag = 2
                elif sys.argv[i+1] == 'T':
                    simflag = 3
                i += 2
            elif iarg == 'plot_raw_data':
                showdata = 1
                i += 1
            elif iarg == 'plot_fit':
                showfit = 1
                i += 1
            else:
                raise ValueError(f"{iarg} is not a valid argument.")
                i += 1
            print(i)
    
    #--------Define directory variables------
    cwd = os.getcwd()
    read_dir = os.path.join(cwd, 'data')
    write_dir = os.path.join(cwd, 'output')
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    #------------Create list of files----------------

    # simflag == 1: different viscosity, same k and T    
    if simflag==1:
        read_dir = os.path.join(read_dir, 'diff_vis')
        files=os.listdir(read_dir)
        invisc = np.array([])
        Tin = np.array([])
        kmap = {'0.065': 2*np.pi/96,'0.071': 2*np.pi/88,'0.079': 2*np.pi/80, '0.098': 2*np.pi/64}
        k = np.array([]) # Required if reading k from the file name
        for file in files:
            invisc = np.append(invisc, float(file.split('_')[2]))
            Tin = np.append(Tin, float(file.split('_')[4]))
            k = np.append(k, kmap[file.split('_')[-1][:-4]]) # For brevity, I used rounded k in file names.
                                                             # kmap maps the k in title to the actual value. 
                                                             # This could be potentially improved by using the box length
                                                             # instead of k in the file names.i

        dtlist = np.ones(len(files))    # Timestep
        dx = 1

    # simflag==2: different k, same T and viscosity
    elif simflag==2:
        read_dir = os.path.join(read_dir, 'diff_k')
        files=os.listdir(read_dir)
        kmap = {'0.065': 2*np.pi/96,'0.071': 2*np.pi/88,'0.079': 2*np.pi/80, '0.098': 2*np.pi/64}
        invisc = np.array([])
        Tin = np.array([])
        k = np.array([]) # Required if reading k from the file name
        for file in files:
            invisc = np.append(invisc, float(file.split('_')[2]))
            Tin = np.append(Tin, float(file.split('_')[4]))
            k = np.append(k, kmap[file.split('_')[-1][:-4]]) # For brevity, I used rounded k in file names.
                                                             # kmap maps the k in title to the actual value. 
                                                             # This could be potentially improved by using the box length
                                                             # instead of k in the file names.
        dtlist = np.ones(len(files))
        dx = 1
    
    # simflag == 3: different T, same k and viscosity
    elif simflag == 3:
        read_dir = os.path.join(read_dir, 'diff_T')
        files=os.listdir(read_dir)
        invisc = np.array([])
        Tin = np.array([])
        kmap = {'0.065': 2*np.pi/96,'0.071': 2*np.pi/88,'0.079': 2*np.pi/80, '0.098': 2*np.pi/64}
        k = np.array([]) # Required if reading k from the file name
        for file in files:
            invisc = np.append(invisc, float(file.split('_')[2]))
            Tin = np.append(Tin, float(file.split('_')[4]))
            k = np.append(k, kmap[file.split('_')[-1][:-4]]) # For brevity, I used rounded k in file names.
                                                             # kmap maps the k in title to the actual value. 
                                                             # This could be potentially improved by using the box length
                                                             # instead of k in the file names.
        
        dtlist = np.ones(len(files))
        dx = 1
    
    #------Cross-simulation arrays------
    viscos = np.array([])
    gamma = np.array([])
    omega = np.array([])
    T=np.array([])
    ampl = np.array([])
    rho = np.array([])
    velo = np.array([])

    i=0 #simulation counter

    #------------Plot initializations------------------
    if showfit==1:
        fig_fit, ax_fit = plt.subplots(figsize=FIG_SIZE)
    if showdata == 1:
        fig_data, ax_data = plt.subplots(figsize=FIG_SIZE)

    #------------Loop over files-----------------------
    for file in files:
        print(f"Analyzing {file}")
        swdf = pd.read_csv(os.path.join(read_dir,file), header=0, names=["rho", "temp","vz","timestep"]) # swdf = sound wave DataFrame
        dt = dtlist[i]
        swdf["t"] = swdf["timestep"]*dt
        T = np.append(T, np.mean(swdf['temp']))
        rho = np.append(rho, np.mean(swdf['rho']))
        if yflag == 1: #density
            fit_params, peaks = fit_wave(swdf['t'], swdf['rho'], wave_type='cosine', wave_init = [0.0001, 0.001,0.0320, 0, 1], peak_init=[0.001, 0.001,1], return_peak=True)
            ampl = np.append(ampl, fit_params[0])
            gamma = np.append(gamma, fit_params[1])
            omega = np.append(omega, fit_params[2])
            viscos = np.append(viscos, compute_viscosity(gamma[i], k[i], rho[i]))
            if showdata == 1:
                ax_data.plot(np.array(swdf["t"]), np.array(swdf["rho"]), label=fr"$\eta = {invisc[i]}$")
                ax_data.set_xlabel(r"time ($\mu s$)", fontsize=15)
                ax_data.set_ylabel(r"$\rho (pg/\mu m^3)$", fontsize=15)
                ax_data.legend(fontsize=14)
                ax_data.tick_params(labelsize=13)
            if showfit==1:
                tpeak = peaks[0]; ypeak = peaks[1]
                ax_fit.plot(tpeak,ypeak, 'ok')
                ax_fit.plot(tpeak, fit_exp(tpeak,*expParam))
                ax_fit.plot(np.array(swdf["t"]), fit_cosine(np.array(swdf["t"]), *param), label=fr"{param[0]:.8f}*exp(-{param[1]:.8f}t)cos({param[2]:.8f}t+{param[3]:.8f})+{param[4]:.1f}")
                ax_fit.set_xlabel(r"time ($\mu s$)")
                ax_fit.set_ylabel(r"$\rho (pg/\mu m^3)$")
                ax_fit.legend()
            i+=1
        elif yflag== 2: #velocity
            fit_params = fit_wave(swdf['t'], swdf['vz'], wave_param = [0.0001, 0.001,0.0320, 0, 0], fit_peak = False)
            ampl = np.append(ampl, fit_params[0])
            gamma = np.append(gamma, fit_params[1])
            omega = np.append(omega, fit_params[2])
            viscos = np.append(viscos, compute_viscosity(gamma[i], k[i], rho[i]))
            if showfit == 1:
                plt.plot(tpeak,ypeak, 'ok')
                plt.plot(tpeak, fit_exp(tpeak,*expParam))
                plt.plot(np.array(swdf["t"]), fit_sine(np.array(swdf["t"]), *param), label=fr"{param[0]:.8f}*exp(-{param[1]:.8f}t)sin({param[2]:.8f}t+{param[3]:.8f})+{param[4]:.1f}")
            plt.xlabel(r"time ($\mu s$)")
            plt.ylabel(r"$v_z (\mu m /\mu s)$")
            plt.legend()
            i+=1
    
        elif yflag == 3: #temperature
            print('The code is not developed for temperature waves. Sorry!')
            exit()
            # We did not study temperature waves for the paper so left this one. But should be straightfoward similar to density and velocity 
    #------------Plotting--------------

    if simflag == 1:# simflag == 1: different viscosity, same k and T
        if saveflag and showfit:
            fig_fit.savefig(os.path.join(write_dir, "soundwaves_fit_diffvis_k0.079.png"), dpi=300, bbox_inches="tight")
        if saveflag and showdata:
            fig_data.savefig(os.path.join(write_dir, "soundwaves_diffvis_k0.079.png"), dpi=300, bbox_inches="tight")

        #-----------gamma vs viscosity--------------
        fig_vis, ax_vis = plt.subplots(figsize= FIG_SIZE)
        coeff = np.polyfit(invisc, gamma, 1) #Fit a line to input viscosities vs. computed gamma
        ax_vis.plot(invisc, gamma, "o")
        ax_vis.plot(np.linspace(invisc[0], invisc[-1],100),np.linspace(invisc[0], invisc[-1],100)*coeff[0]+coeff[1], label=fr"{coeff[0]:0.7f} $\eta$ + {abs(coeff[1]):0.1f}" )
        ax_vis.set_xlabel(r"$\eta (\frac{pg}{\mu m \mu s})$")
        ax_vis.set_ylabel(r"$\gamma (1/\mu s)$")
        ax_vis.legend()
        if saveflag==1:
            fig_vis.savefig(os.path.join(write_dir, "damp_vs_vis.png"), dpi=300, bbox_inches="tight")
        
        #----------To store wave information to a file---------------
        #omega_df = pd.DataFrame({'rho':np.round(rho, 7), 'vis':invisc, 'k':np.round(k, 7), 'T':np.round(T, 7), 'omega':np.round(omega, 7), 'gamma':np.round(gamma, 7), 'ampl':np.abs(np.round(ampl, 7))})
        #omega_df.to_csv(os.path.join(write_dir, f'vis_variable_k{k[0]:0.3}T{T[0]:0.3}.csv'), index=False)

    elif simflag == 2:# simflag==2: different k, same T and viscosity
        if saveflag and showfit:
            fig_fit.savefig(os.path.join(write_dir, "soundwaves_fit_diff_k.png"), dpi=300, bbox_inches="tight")
        if saveflag and showdata:
            fig_data.savefig(os.path.join(write_dir, "soundwaves_raw_diff_k.png"), dpi=300, bbox_inches="tight")

        # ----------dispersion vs k^2, linear fit-----------
        colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        markers = ['o', 'D', 's', '*']
        fig_k, ax_k = plt.subplots()
        p_counter = 0
        for T_choice in np.unique(Tin):
            gamma_choice = np.array([])
            omega_choice = np.array([])
            k_choice = np.array([])
            for i in range(len(files)):
                if np.abs(T[i]- T_choice) < 1e-5:
                    gamma_choice = np.append(gamma_choice, gamma[i])
                    omega_choice = np.append(omega_choice, omega[i])
                    k_choice = np.append(k_choice, k[i])
            if len(k_choice) < 2:
                print(f"Only one point for {T_choice}. Cannot fit a line. Skip!")
                continue
            ax_k.plot(k_choice**2, omega_choice**2+gamma_choice**2, color=colours[p_counter], marker=markers[p_counter])
            paramO = np.polyfit(k_choice**2, omega_choice**2+gamma_choice**2, 1)
            ax_k.plot(np.linspace(k_choice[0]**2, k_choice[-1]**2, 100), np.linspace(k_choice[0]**2, k_choice[-1]**2, 100)*paramO[0]+paramO[1],
                      color = colours[p_counter],label=rf"{paramO[0]:0.3f} $k^2$")
            p_counter += 1
        ax_k.set_xlabel(r"$k^2(\frac{1}{\mu m^2})$")
        ax_k.set_ylabel(r"$\omega^2+\gamma^2 (1/\mu s^2)$")
        ax_k.legend()
        if saveflag == 1:
            fig_k.savefig(os.path.join(write_dir, f"dispersion_vs_k2_vis{np.mean(viscos):0.3}.png"), dpi=300, bbox_inches="tight")

        #----------To store wave information to a file---------------
        #omega_df = pd.DataFrame({'rho':np.round(rho, 7), 'vis':invisc, 'k':np.round(k, 7), 'T':np.round(T, 7), 'omega':np.round(omega, 7), 'gamma':np.round(gamma, 7), 'ampl':np.abs(np.round(ampl, 7))})
        #omega_df.to_csv(os.path.join(write_dir, f'k_variable_vis{np.mean(viscos):0.2}T{np.mean(T):0.2}.csv'), index=False)

    elif simflag == 3:# simflag == 3: different T, same k and viscosity
        if saveflag and showfit:
            fig_fit.savefig(os.path.join(write_dir, "soundwaves_fit_diff_T.png"), dpi=300, bbox_inches="tight")
        if saveflag and showdata:
            fig_data.savefig(os.path.join(write_dir, "soundwaves_raw_diff_T.png"), dpi=300, bbox_inches="tight")

        #-----Dispersion Relation for different T--------
        fig_temp, ax_temp = plt.subplots(figsize= FIG_SIZE)
        coeff = np.polyfit(T, omega**2+gamma**2, 1)
        ax_temp.plot(T, omega**2+gamma**2, "o", markersize=6)
        ax_temp.plot(np.linspace(np.amin(T), np.amax(T), 100), np.linspace(np.amin(T), np.amax(T), 100)*coeff[0]+coeff[1], label=rf"{coeff[0]:0.6f} T")
        ax_temp.set_xlabel(r"$T(\frac{\mu m^2}{\mu s^2})$", fontsize=15)
        ax_temp.set_ylabel(r"$\omega^2+\gamma^2 (1/\mu s^2)$", fontsize=15)
        ax_temp.tick_params(labelsize=13)
        ax_temp.legend(fontsize=14)
        if saveflag ==1:
            plt.savefig(os.path.join(write_dir, f"dispersion_diff_T_vis{np.mean(viscos):0.3}.png"), dpi=300, bbox_inches="tight")


    
    if saveflag==0:
        plt.show()
