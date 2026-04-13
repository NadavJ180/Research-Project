import numpy as np  
import matplotlib.pyplot as plt 

# Constants
h = 6.62607015e-34  # Planck's constant [J*s]
kB = 1.380649e-23    # Boltzmann's constant [J/K]
w = 1e13            # Frequency of the harmonic oscillator [Hz]

# Temperature range
T = np.linspace(1, 1000, 100)  # Kelvin

def Cv_HO(T, h, kB, w):
    f = h * w / (kB * T)
    return f**2 * np.exp(f) / (np.exp(f) - 1)**2

def Cv_Box(T, h, kB, w):
    return 

# HO Heat Capacity
Cv_HO_val = Cv_HO(T, h, k, w)

# Box Heat Capacity


# Plot the results HO
plt.figure(figsize=(10, 6))
plt.plot(T, Cv_HO_val, label='Heat Capacity')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/K)')
plt.title('Heat Capacity of a Harmonic Oscillator')
plt.legend()
plt.grid(True)
plt.show()  

