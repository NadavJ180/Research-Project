import numpy as np
import matplotlib.pyplot as plt

# Constants (normalized so kB = 1 for simplicity)
kB = 1.0

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
        E_n = n**2
        beta = 1.0 / T
        
        # Partition function and its derivatives
        exp = np.exp(-beta * E_n)
        z = np.sum(exp)
        E_ave = np.sum(E_n * exp) / z
        E2_ave = np.sum(E_n**2 * exp) / z
        
        # Cv = ( <E^2> - <E>^2 ) / (kB * T^2)
        Cv = (E2_ave - E_ave**2) * beta**2 * kB
        Cv_list.append(Cv)
    return np.array(Cv_list)

#Number of Box Potential Energy Levels
n_levels = [2,10,20,100]

# Temperature range (log scale)
T_range = np.logspace(-2, 2, 400)

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
