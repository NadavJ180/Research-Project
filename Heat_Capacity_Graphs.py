import numpy as np  
import matplotlib.pyplot as plt 

# Constants
kB = 1  # Boltzmann's constant (dimensionless)

# Temperature range
T = np.linspace(0.02, 5, 300)  # Kelvin

# --- HO Heat Capacity ---

def Cv_HO(T, hw=1.0):
    x = hw / (kB * T)
    return kB * (x**2 * np.exp(x)) / (np.exp(x) - 1)**2

# --- Box Heat Capacity ---

def Cv_box(T, N):
    Cv = []

    n = np.arange(1, N+1) # number of energy levels
    En = n**2 # energy levels (constants are set to 1)

    En = En - En[0]  # shift energies so that the ground state is at zero

    Cv = np.zeros_like(T)
    
    for i, temp in enumerate(T):

        weights = np.exp(-En / (kB * temp))

        Z = np.sum(weights)  # partition function
        E_ave = np.sum(En * weights) / Z  
        E_2_ave = np.sum(En**2 * weights) / Z  
    
        Cv[i] = (E_2_ave - E_ave**2) / (kB * temp**2)

    return Cv

# --- num of Box levels ---
N_vals = [2, 3, 5, 10, 20, 50, 200]

# --- Compute heat capacities ---
plt.figure

for N in N_vals:
    Cv_vals = Cv_box(T, N)
    plt.plot(T, Cv_vals, label=f'N={N}')

plt.xlabel('Temperature (dimensionless)')
plt.ylabel('Heat Capacity (dimensionless)')
plt.title('Effect of Number of Energy Levels on Heat Capacity (Box Potential)')
plt.legend()
plt.grid()

plt.show()

'''
Cv_HO_val = Cv_HO(T)
Cv_box_val = Cv_box(T)

# ---Plot ---
plt.figure()
plt.plot(T, Cv_HO_val, label='Harmonic Oscillator', color='blue')
plt.plot(T, Cv_box_val, label='Particle in a Box', color='orange')

plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (C_v)')
plt.title('Heat Capacity vs Temperature')

plt.legend()
plt.grid()

plt.show()
'''
