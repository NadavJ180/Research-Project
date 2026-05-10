import numpy as np
import matplotlib.pyplot as plt

# Constants (normalized so kB = 1 for simplicity)
kB = 1.0
m = 1.0
L = 1.0
h_bar = 0.1
E_g = (h_bar**2 * np.pi**2) / (2 * m * L**2)  # Ground state energy for box potential

# --- Harmonic Oscillator Heat Capacity ---

def Cv_HO(T, hw=1.0):
    x = hw / (kB * T)
    return kB * (x**2 * np.exp(x)) / (np.exp(x) - 1)**2

# --- Box Potential Heat Capacity ---

def Cv_Box_Quantum(temp_ratio, n_levels):
    # temp_ratio = kB * T / Eg
    # En = n^2 * Eg
    Cv_list = []
    for T in temp_ratio:
        n = np.arange(1, n_levels + 1)
        E_n = n**2 * E_g
        beta = 1.0 / (T * kB)
        
        # Partition function and its derivatives
        exp = np.exp(-beta * E_n)
        z = np.sum(exp)
        E_ave = np.sum(E_n * exp) / z
        E2_ave = np.sum(E_n**2 * exp) / z
        
        # Cv = ( <E^2> - <E>^2 ) / (kB * T^2)
        Cv = (E2_ave - E_ave**2) * beta**2 * kB
        Cv_list.append(Cv)
    return np.array(Cv_list)

def Cv_Box_Classical(temp_ratio, n_levels, xi):
    # temp_ratio = kB * T / Eg
    # En = n^2 * Eg
    Cv_list = []
    for T in temp_ratio:
        n = np.arange(1, n_levels + 1)
        E_n = n**2 * E_g / xi**2
        beta = 1.0 / (T * kB)
        
        #Log-sum-exp trick for numerical stability (stop 1/0 overflow)
        max_exponent = np.max(-beta * E_n)    
        exp_shift = np.exp(-beta * E_n - max_exponent)
        
        # Partition function and its derivatives
        z = np.sum(exp_shift)
        E_ave = np.sum(E_n * exp_shift) / z
        E2_ave = np.sum(E_n**2 * exp_shift) / z
        
        # Cv = ( <E^2> - <E>^2 ) / (kB * T^2)
        Cv = (E2_ave - E_ave**2) * beta**2 * kB

        Cv_list.append(Cv)
    return np.array(Cv_list)

def find_classical_limit_Box(temp_ratio, n_levels, tolerance=1e-4):
    
    xi = 1.0
    converged = False

    current_Cv = Cv_Box_Classical(temp_ratio, n_levels, xi)

    while not converged:
        xi_next = xi * 2
        next_Cv = Cv_Box_Classical(temp_ratio, n_levels, xi_next)
        # Check for convergence
        
        rmse = np.sqrt(np.mean((next_Cv - current_Cv)**2))

        if rmse < tolerance:
            converged = True
        else:
            xi = xi_next
            current_Cv = next_Cv
        
        
        if xi > 1e2:  # limit xi because of finite n_levels
            break   

    return current_Cv, xi

#Number of Box Potential Energy Levels
n_levels = [2,10,20,100]
n = 25000

# Temperature range (log scale)
T_range = np.logspace(-2, 2, 400)

# Create Classical limit plot for the box potential
fig = plt.figure(figsize=(6, 5))
classical_limit_box, xi_converged = find_classical_limit_Box(T_range, n)
plt.plot(T_range, classical_limit_box, '--', color='yellowgreen', label=f'Classical Limit (xi={xi_converged:.2f})')
plt.title("Classical Limit for Box Potential")
plt.xlabel(r"$k_B T / E_g$")
plt.ylabel(r"$C_v [J/K]$")
plt.xscale('log')
plt.ylim(0, 0.7)
plt.legend()
plt.tight_layout()
plt.show()

'''
# Create the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Left Plot: Harmonic Potential ---
ax1.plot(T_range, [kB]*len(T_range), '--', color='yellowgreen', label='Classical')
ax1.plot(T_range, Cv_HO(T_range), '-', color='steelblue', label='Quantum')
ax1.set_title("Harmonic potential")
ax1.set_xlabel(r"$k_B T / E_g$")
ax1.set_ylabel(r"$C_v [J/K]$")
ax1.set_xscale('log')
ax1.set_ylim(0, 1.2)
ax1.legend()

# --- Right Plot: Box Potential ---
ax2.plot(T_range, [0.5*kB]*len(T_range), '--', color='yellowgreen', label='Classical')
for n in n_levels:
    ax2.plot(T_range, Cv_Box_Quantum(T_range, n), '-', label=f'Quantum (n={n})')
ax2.set_title("Box potential")
ax2.set_xlabel(r"$k_B T / E_g$")
ax2.set_ylabel(r"$C_v [J/K]$")
ax2.set_xscale('log')
ax2.set_ylim(0, 0.7)
ax2.legend()

plt.tight_layout()
plt.show()
'''