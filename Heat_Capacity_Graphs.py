import numpy as np  
import matplotlib.pyplot as plt 

# Constants
h = 6.62607015e-34  # Planck's constant (J*s)
k = 1.380649e-23    # Boltzmann's constant (J/K)
v = 1e13            # Frequency of the harmonic oscillator (Hz)

# Temperature range
T = np.linspace(1, 1000, 100)  # Kelvin

def Cv_HO(T, h, k, v):
    return (h * v / (k * T))**2 * np.exp(h * v / (k * T)) / (np.exp(h * v / (k * T)) - 1)**2

def Cv_Box(T, h, k, v):
    return 

# HO Heat Capacity
Cv_HO_val = Cv_HO(T, h, k, v)

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

