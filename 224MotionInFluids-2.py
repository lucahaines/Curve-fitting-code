#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:28:17 2024

@author: lucahaines
"""
# Free Choice Experiment 3
# Motion in Fluids
# Luca Haines and Noah Vaillant

# =============================================================================

# Importing useful modules

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp

# =============================================================================

# Introducing variables

# Bead diameters and radii (all in mm)
tef_beads_d = [(1.59,0.02), (2.37,0.02), (3.16,0.02), (4.73,0.02), (6.34,0.02)]
tef_beads_r = [(1.59/2,0.02/2), (2.37/2,0.02/2), (3.16/2,0.02/2), (4.73/2,0.02/2), 
               (6.34/2,0.02/2)]
nyl_beads_d = [(2.36,0.01), (3.15,0.01), (3.93,0.02),(4.74,0.02), (6.32,0.02)]
nyl_beads_r =  [(2.36/2,0.01/2), (3.15/2,0.01/2), (3.93/2,0.02/2), (4.74/2,0.02/2) ,
                (6.32/2,0.02/2)]

# All at 25 degrees celsius

# Glycerol density 
ρg = 1260 # kg/m−3

# Glycerol viscosity 
ηg = 0.934 # kg/ms

# Water viscosity 
ηw = 0.001 # kg/ms

# Water density 
ρw = 1000 # kg/m−3

# Teflon density 
ρt = 2200 # kg/m−3

# Nylon density 
ρn = 1120 # kg/m−3

# pi
pi = np.pi

# Gravity
g = 9.81

# Diameter of tank
D = 0.09402

# =============================================================================

# Functions

# Raw data to time

def time(x):
    t = x[:,0]
    return t


# Raw data to position

def position(x):
    p =  (1/1000) * x[:,1] # Convert mm to m
    return p


# Remove values where software malfunctioned and measured position to be 0

def remove(t, p):
    indexes = []
    for ii in range(len(p)):
        if p[ii] == 0:
            indexes.append(ii)
    t_new = np.delete(t, indexes)
    p_new = np.delete(p, indexes)
    return t_new, p_new


# Extract data

def data_extract(a): 
    d = np.loadtxt(a, comments='#', delimiter='\t', skiprows=2)
    return remove(time(d), position(d))


# Plot data

def plot_data(a):
    c = data_extract(a)
    plt.figure(figsize = (10, 6))
    plt.grid()
    plt.title(a)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.plot(c[0], c[1])
    
    
# Curve Fitting to find Velocity (Uncorrected)

def f(a, t, c):
    return a * t + c


# Finding Velocity for a given drop

def find_velocity(a):
    c = data_extract(a)
    t = c[0]
    y = c[1]
    
    # Handle data truncation for large beads in glycerine
    if '4GLC' in a or '5GLC' in a:
        t = c[0][int(len(c[0])/2):]
        y = c[1][int(len(c[1])/2):]
    
    # Setting uncertainties for water trials
    if 'WAT' in a:
        if a in ['1WAT1.txt', '1WAT2.txt']:
            yunc = np.ones(len(y)) * tef_beads_d[0][0] * (1/1000)
        elif a in ['2WAT1.txt', '2WAT2.txt']:
            yunc = np.ones(len(y)) * tef_beads_d[1][0] * (1/1000)
        elif a in ['3WAT1.txt', '3WAT2.txt']:
            yunc = np.ones(len(y)) * tef_beads_d[2][0] * (1/1000)
        elif a in ['4WAT1.txt', '4WAT2.txt']:
            yunc = np.ones(len(y)) * tef_beads_d[3][0] * (1/1000)
        elif a in ['5WAT1.txt', '5WAT2.txt']:
            yunc = np.ones(len(y)) * tef_beads_d[4][0] * (1/1000)
    
    # Setting uncertainties for glycerine trials
    elif 'GLC' in a:
        if a in ['1GLC1.txt', '1GLC2.txt']:
            yunc = np.ones(len(y)) * nyl_beads_d[0][0] * 1000**-1
        elif a in ['2GLC1.txt', '2GLC2.txt']:
            yunc = np.ones(len(y)) * nyl_beads_d[1][0] * 1000**-1
        elif a in ['3GLC1.txt', '3GLC2.txt']:
            yunc = np.ones(len(y)) * nyl_beads_d[2][0] * 1000**-1
        elif a in ['4GLC1.txt', '4GLC2.txt']:
            yunc = np.ones(len(y)) * nyl_beads_d[3][0] * 1000**-1
        elif a in ['5GLC1.txt', '5GLC2.txt']:
            yunc = np.ones(len(y)) * nyl_beads_d[4][0] * 1000**-1
    
    # Perform curve fit
    popt, pcov = sp.curve_fit(f, t, y, sigma=yunc)
    unc = np.sqrt(np.diag(pcov))
    
    return [popt, pcov, unc, yunc]

# Finding additional uncertainty (different in vterm for both trials)
def additional_vunc(a, b):
    return (abs(find_velocity(a)[0][0] - find_velocity(b)[0][0]))/2


# Plotting fitted curve for a given drop

def plot_fit(a):
    c = data_extract(a)
    t = c[0]
    y = c[1]
    popt, pcov, unc, yunc = find_velocity(a)
    if '4GLC' in a or '5GLC' in a:
        t = c[0][int(len(c[0])/2):]
        y = c[1][int(len(c[1])/2):]
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.errorbar(t, y, yerr = yunc, label="Data", color="blue")
    plt.plot(t, f(t, *popt),label = 'Curve Fit', color="red")
    plt.title("Position vs. Time with Fitted Curve" + ' (' +  a[:-4] + ')')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.show()
    print('The Velocity was found to be', round(popt[0], 5),'+/-', round(unc[0], 5), 'm/s ')


# Correction equation for boudary effects of container, only effects water drops
# Not used in our final analysis

def v_correction(d, vm):
    vcorr = vm * (1 - 2.104 * (d/D) + 2.089 * (d/D)**3)**-1
    
    return vcorr


# Calculate Reynolds number for a given drop

def calculate_reynolds(a):
    v = find_velocity(a)[0][0]
    
    # Finding characteristic length (diameter) for water trials
    if 'WAT' in a:
        if a == '1WAT1.txt' or a == '1WAT2.txt':
            d = nyl_beads_d[0][0] * 1/1000
        elif a == '2WAT1.txt' or a == '2WAT2.txt':
            d = nyl_beads_d[1][0] * 1/1000
        elif a == '3WAT1.txt' or a == '3WAT2.txt':
            d = nyl_beads_d[2][0] * 1/1000
        elif a == '4WAT1.txt' or a == '4WAT2.txt':
            d = nyl_beads_d[3][0] * 1/1000
        elif a == '5WAT1.txt' or a == '5WAT2.txt':
            d = nyl_beads_d[4][0] * 1/1000
    
    # Finding characteristic length (diameter) for Glycerine trials
    else:
        if a == '1GLC1.txt' or a == '1GLC2.txt':
            d = tef_beads_d[0][0] * 1/1000
        elif a == '2GLC1.txt' or a == '2GLC2.txt':
            d = tef_beads_d[1][0] * 1/1000
        elif a == '3GLC1.txt' or a == '3GLC2.txt':
            d = tef_beads_d[2][0] * 1/1000
        elif a == '4GLC1.txt' or a == '4GLC2.txt':
            d = tef_beads_d[3][0] * 1/1000
        elif a == '5GLC1.txt' or a == '5GLC2.txt':
            d = tef_beads_d[4][0] * 1/1000
    
    
    # Finding viscosity
    if 'WAT' in a:
        vis = ηw
    elif 'GLC' in a:
        vis = ηg
        
    # Finding density
    if 'WAT' in a:
        dens = ρw
    elif 'GLC' in a:
        dens = ρg
    
    # Return Reynolds numbers
    
    if 'WAT' in a:
        return [(dens * d * v)/vis, d/2] # Also returns the radius for given a
    elif 'GLC' in a:
        return [(dens * d * v)/vis, d/2]


# Define functions for both high and low reynolds numbers

def high_reynolds_function(r, a, c):
    return a * r**0.5 + c

def low_reynolds_function(r, b, c):
    return b * r**2 + c


# Collecting all data

def collect_water_data():
    wat_files = ['1WAT1.txt', '1WAT2.txt', '2WAT1.txt', '2WAT2.txt', 
                   '3WAT1.txt', '3WAT2.txt', '4WAT1.txt', '4WAT2.txt',
                   '5WAT1.txt', '5WAT2.txt']
    velocities = []
    vunc = []
    radii = []
    
    additional_unc = []
    additional_unc.append(additional_vunc('1WAT1.txt', '1WAT2.txt'))
    additional_unc.append(additional_vunc('2WAT1.txt', '2WAT2.txt'))
    additional_unc.append(additional_vunc('3WAT1.txt', '3WAT2.txt'))
    additional_unc.append(additional_vunc('4WAT1.txt', '4WAT2.txt'))
    additional_unc.append(additional_vunc('5WAT1.txt', '5WAT2.txt'))
                          
    
    for a in wat_files:
            v = find_velocity(a)[0][0]
            v_unc = find_velocity(a)[2][0]
            r = calculate_reynolds(a)[1]
            velocities.append(v)
            vunc.append(v_unc)
            radii.append(r)
    
    # Making new list of nylon bead diams to account for two drops per diam
    new_nyl = []
    for ii  in range(len(nyl_beads_d)):
        new_nyl.append(nyl_beads_d[ii][0]*1/1000)
        new_nyl.append(nyl_beads_d[ii][0]*1/1000)
    
    corrected_velocities = []
    for i in range(len(velocities)):
          corrected_velocities.append(v_correction(new_nyl[i], velocities[i]))
    
    
    new_radii = []
    new_vunc = []
    new_velocities = []
    new_velocities_corr = []
            
    for i in range(len(radii)):
        if i % 2 == 0:
            new_radii.append((radii[i] + radii[i+1])/2)
            
    for i in range(len(vunc)):
        if i % 2 == 0:
            new_vunc.append((vunc[i] + vunc[i+1])/2)
    
    for i in range(len(velocities)):
        if i % 2 == 0:
            new_velocities.append((velocities[i] + velocities[i+1])/2)
    
    for i in range(len(new_vunc)):
        new_vunc[i] = np.sqrt(new_vunc[i]**2 + additional_unc[i]**2)
        
    for i in range(len(velocities)):
        if i % 2 == 0:
            new_velocities_corr.append((corrected_velocities[i] + corrected_velocities[i+1])/2)
        
    return np.array(new_radii),np.array(new_velocities),np.array(new_vunc), np.array(new_velocities_corr)

# Function to collect all velocities and radii for glycerine trials
def collect_glycerine_data():
    glc_files = ['1GLC1.txt', '1GLC2.txt', 
                 '2GLC1.txt', '2GLC2.txt',
                 '3GLC1.txt', '3GLC2.txt', '4GLC1.txt', '4GLC2.txt',
                 '5GLC1.txt', '5GLC2.txt']
    velocities = []
    vunc = []
    radii = []
    
    # Finding additional uncertainties in velocity
    additional_unc = []
    additional_unc.append(additional_vunc('1GLC1.txt', '1GLC2.txt'))
    additional_unc.append(additional_vunc('2GLC1.txt', '2GLC2.txt'))
    additional_unc.append(additional_vunc('3GLC1.txt', '3GLC2.txt'))
    additional_unc.append(additional_vunc('4GLC1.txt', '4GLC2.txt'))
    additional_unc.append(additional_vunc('5GLC1.txt', '5GLC2.txt'))
    
    for a in glc_files:
        
            v = find_velocity(a)[0][0]
            v_unc = find_velocity(a)[2][0]
            r = calculate_reynolds(a)[1]
            velocities.append(v)
            vunc.append(v_unc)
            radii.append(r)
            
    new_tef = []
    for ii  in range(len(tef_beads_d)):
        new_tef.append(tef_beads_d[ii][0]*1/1000)
        new_tef.append(tef_beads_d[ii][0]*1/1000)
    
    corrected_velocities = []
    for i in range(len(velocities)):
          corrected_velocities.append(v_correction(new_tef[i], velocities[i]))
        
        
    new_radii = []
    new_vunc = []
    new_velocities = []
    new_velocities_corr = []
            
    for i in range(len(radii)):
        if i % 2 == 0:
            new_radii.append((radii[i] + radii[i+1])/2)
            
    for i in range(len(vunc)):
        if i % 2 == 0:
            new_vunc.append(np.sqrt(vunc[i]**2 + vunc[i+1]**2))
    
    for i in range(len(velocities)):
        if i % 2 == 0:
            new_velocities.append((velocities[i] + velocities[i+1])/2)
            
    for i in range(len(velocities)):
        if i % 2 == 0:
            new_velocities_corr.append((corrected_velocities[i] + corrected_velocities[i+1])/2)
            
    for i in range(len(new_vunc)):
        new_vunc[i] = np.sqrt(new_vunc[i]**2 + additional_unc[i]**2)
        
        
    return np.array(new_radii),np.array(new_velocities),np.array(new_vunc), np.array(new_velocities_corr)


# Uncertainty for corrected velocity (water regime)
def correction_unc():
    vunc = collect_water_data()[2]
    v = collect_water_data()[1]
    dunc = []
    for i in range(len(tef_beads_d)):
        dunc.append(tef_beads_d[i][1])
    Dunc = 0.01
    
    vcorr_unc = []
    for i in range(len(vunc)):
        d = nyl_beads_d[i][1]
        vm = v[i]
        vcorr_unc.append(np.sqrt((1/(((1-2.104*d/D + 2.089*(d/D)**3))**2)*vunc[i]**2 +
                                 ((2.104 * D**5 * vm - 6.267*D**3*vm*d**2)/
                                 (D**3-2.104*D**2*d+2.089*d**3)**2)*dunc[i]**2
                                 +(6.267*d**3*vm*D**2-2.104*d*vm*D**4)/
                                 (2.089*d**3 - 2.104*d*D**2 + D**3)**2*Dunc**2)))
    
    return vcorr_unc
    

# Curve fitting functions
def curve_fit_high():
    
    # Get water data
    # corr_unc = correction_unc()
    r = collect_water_data()[0]
    v = (collect_water_data()[1])
    # v_corr = (collect_water_data()[3])
    vunc = collect_water_data()[2]
    
    # unc_corr = []
    # for ii in range(len(vunc)):
    #     unc_corr.append(np.sqrt((2*vunc[ii]/3)**2 + (corr_unc[ii]/3)**2))
    
    
    # Making the plots smoother
    r_smooth = np.linspace(min(r), max(r), 1000)
    
    # Execute curve fit
    popt, pcov = sp.curve_fit(high_reynolds_function, r, v, sigma=vunc)
    unc = np.sqrt(np.diag(pcov))
    
    # popt_corr, pcov_corr = sp.curve_fit(high_reynolds_function, r, v_corr, sigma=unc_corr)
    # unc_c = np.sqrt(np.diag(pcov_corr))
    
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(r, v, yerr=vunc, fmt='o', label='Data, no correction')
    plt.plot(r_smooth, high_reynolds_function(r_smooth, *popt), 'r-', label='Fit, no correction')
    #plt.errorbar(r, v_corr, yerr=unc_corr, fmt='o', label='Data, correction')
    #plt.plot(r_smooth, high_reynolds_function(r_smooth, *popt_corr), 'r-', label='Fit, correction')
    plt.xlabel('Radius (m)')
    plt.ylabel('Terminal Velocity (m/s)')
    plt.title('High Reynolds Number Regime (Water)')
    plt.plot(r_smooth, (((8*(ρn-ρw)*g)/(3*ρw*0.56)))**0.5 * r_smooth**0.5, label = 'Theoretical')
    plt.grid()
    plt.legend()
    
    # Print parameters
    print(f"Fit parameters: a = {popt[0]:.3f} ± {unc[0]:.3f}, c = {popt[1]:.3f} ± {unc[1]:.3f}")
    
    return popt, unc

def curve_fit_low():
    
    # Get glycerine data
    corr_unc = correction_unc()
    r = collect_glycerine_data()[0]
    v = collect_glycerine_data()[1]
    v_corr = collect_glycerine_data()[3]
    vunc = collect_glycerine_data()[2]
    
    unc_corr = []
    for ii in range(len(vunc)):
        unc_corr.append(np.sqrt((2*vunc[ii]/3)**2 + (corr_unc[ii]/3)**2))
        
    
    # Making the plots smoother
    r_smooth = np.linspace(min(r), max(r), 1000)
    
    # Execute curve fit
    popt, pcov = sp.curve_fit(low_reynolds_function, r, v, sigma=vunc)
    unc = np.sqrt(np.diag(pcov))
    
    popt_corr, pcov_corr = sp.curve_fit(high_reynolds_function, r, v_corr, sigma=unc_corr)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(r, v, yerr=vunc, fmt='o', label='Data, no correction')
    plt.plot(r_smooth, low_reynolds_function(r_smooth, *popt), 'r-', label='Fit, no correction')
    #plt.errorbar(r, v_corr, yerr=unc_corr, fmt='o', label='Data, correction')
    # plt.plot(r_smooth, low_reynolds_function(r_smooth, *popt_corr), 'b-', label='Fit, correction')
    plt.xlabel('Radius (m)')
    plt.ylabel('Terminal Velocity (m/s)')
    plt.title('Low Reynolds Number Regime (Glycerine)')
    plt.grid()
    plt.plot(r_smooth, (2/(9*0.934))*(ρt-ρg)*g*r_smooth**2, label = 'Theoretical')

    plt.legend()
    
    # Print parameters
    print(f"Fit parameters: b = {popt[0]:.3f} ± {unc[0]:.3f}, c = {popt[1]:.3f} ± {unc[1]:.3f}")
    
    return popt, unc

# Defining Chi-squared function

def chi_squared(yprediction, ydata, err):
    return sum((yprediction-ydata)**2/err**2)

# Defining reduced Chi-squared function

def chi_red(yprediction, ydata, err, npram):
    return chi_squared(yprediction, ydata, err)/(len(ydata)-npram)

# Finding Chi Values
    
r1 = collect_water_data()[0]
v1 = collect_water_data()[1]
vunc1 = collect_water_data()[2]
for i in range(len(vunc1)):
    vunc1[i] = vunc1[i]
    
r2 = collect_glycerine_data()[0]
v2 = collect_glycerine_data()[1]
vunc2 = collect_glycerine_data()[2]
    
# Curve fits
popt1, pcov1 = sp.curve_fit(high_reynolds_function, r1, v1, sigma=vunc1)
unc1 = np.sqrt(np.diag(pcov1))
popt2, pcov2 = sp.curve_fit(low_reynolds_function, r2, v2, sigma=vunc2)
unc2 = np.sqrt(np.diag(pcov2))
    
chi_water = chi_red(high_reynolds_function(r1, *popt1), v1, vunc1, 2)
chi_glycerine = chi_red(low_reynolds_function(r2, *popt2), v2, vunc2, 2)

print(chi_water)
print(chi_glycerine)

#Plotting the residuals for High Reynolds Number
plt.figure(figsize = (10, 6))
residuals1 = high_reynolds_function(r1, *popt1) - v1

plt.scatter(r1, residuals1)
plt.title('Residuals from High Reynolds Regime')
plt.ylabel('Residual (m/s)')
plt.xlabel('Radius (m)')
plt.grid()
plt.show()

# Plotting the residuals for Low Reynolds Number
plt.figure(figsize = (10, 6))
residuals2 = low_reynolds_function(r2, *popt2) - v2

plt.scatter(r2, residuals2)
plt.title('Residuals from Low Reynolds Regime')
plt.ylabel('Residual (m/s)')
plt.xlabel('Radius (m)')
plt.grid()
plt.show()

# Print off all of the position vs times and velocitie vs times

for a in ['1WAT1.txt', '1WAT2.txt', '2WAT1.txt', '2WAT2.txt', '3WAT1.txt', 
          '3WAT2.txt', '4WAT1.txt', '4WAT2.txt','5WAT1.txt', '5WAT2.txt', 
          '1GLC1.txt', '1GLC2.txt', '2GLC1.txt', '2GLC2.txt','3GLC1.txt', 
          '3GLC2.txt', '4GLC1.txt', '4GLC2.txt','5GLC1.txt', '5GLC2.txt']:
    print('Information about', a)
    print('Position vs time plot for', a)
    print(plot_data(a))
    print('Fitted plot of position vs time', a)
    print(plot_fit(a))
    print('Velocity of', a)
    print(find_velocity(a)[0][0])
    print('Reynolds number of', a)
    print(calculate_reynolds(a))

curve_fit_high()
curve_fit_low()
    
#Plotting the residuals for High Reynolds Number
plt.figure(figsize = (10, 6))
residuals1 = high_reynolds_function(r1, *popt1) - v1

plt.scatter(r1, residuals1)
plt.title('Residuals from High Reynolds Regime')
plt.ylabel('Residual (m/s)')
plt.xlabel('Radius (m)')
plt.grid()
plt.show()

# Plotting the residuals for Low Reynolds Number
plt.figure(figsize = (10, 6))
residuals2 = low_reynolds_function(r2, *popt2) - v2

plt.scatter(r2, residuals2)
plt.title('Residuals from Low Reynolds Regime')
plt.ylabel('Residual (m/s)')
plt.xlabel('Radius (m)')
plt.grid()
plt.show()
