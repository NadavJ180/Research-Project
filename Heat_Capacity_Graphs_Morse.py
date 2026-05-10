import numpy as np
import matplotlib.pyplot as plt


def Cv_Morse(T_range, x_e, h_bar=1, w_e=1, kB=1): #bottom temp limit is set at 1e^-5 [K] 

# ------ max energy levels in accordance with the anharmonic constant ------ 
    n_max = int(np.floor(1 / (2 * x_e) - 0.5))
    n = np.arange(0, n_max + 1)

# ------ energy levels of the Morse potential ------
    E_n = h_bar * w_e * (n + 0.5) - h_bar * w_e * x_e * (n + 0.5) ** 2

    
    Cv_list = []

    for T in T_range:
        
        if T <= 1e-5:  # Avoid numerical issues at very low temperatures
            Cv_list.append(0)
            continue
    
        beta = 1 / (kB * T)
        exp_factors = np.exp(-beta * E_n)
        Z = np.sum(exp_factors) #partition function
        E_avg = np.sum(E_n * exp_factors) / Z #1st moment
        E2_avg = np.sum(E_n**2 * exp_factors) / Z #2nd moment

    # ----- heat capacity Cv calculation -----
        Cv = (E2_avg - E_avg**2) * beta**2 * kB
        Cv_list.append(Cv)

    return Cv_list

# ----- function to plot Cv vs T for a given x_e -----
def plot_Cv_Morse(T_range: np.ndarray, x_e: float):
    Cv_values = Cv_Morse(T_range, x_e)
    plt.figure(figsize=(10, 6))
    plt.plot(T_range, Cv_values, label=f'x_e = {x_e}', color='blue')
    plt.axhline(1, color='red', linestyle='--', label='Classical Limit (Cv = 1)')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Heat Capacity (Cv)')
    plt.title('Heat Capacity vs Temperature for Morse Potential')
    plt.legend()
    plt.grid()
    plt.xscale('log')  # Logarithmic scale for better visibility at low temperatures
    plt.ylim(0, max(Cv_values) * 1.1)  # Set y-axis limit for better visualization
    plt.show()

# ----- function to plot Cv vs T for a list of given x_e -----
def plot_Cv_Morse_Xe_list(T_range: np.ndarray, x_e_list: np.ndarray):
    plt.figure(figsize=(10, 6))

    for x_e in x_e_list:
        Cv_values = Cv_Morse(T_range, x_e)
        plt.plot(T_range, Cv_values, label=f'$x_e$ = {x_e}') 

    plt.axhline(1, color='black', linestyle=':', label='Classical Limit (Cv = 1)')
    plt.xlabel(r"$k_B T / E_g$")
    plt.ylabel(r"$C_v [J/K]$")
    plt.title('Heat Capacity vs Temperature for Morse Potential with Different $x_e$ Values')
    plt.legend()
    plt.grid()
    plt.xscale('log')  # Logarithmic scale for better visibility at low temperatures
    plt.ylim(0, max(Cv_values) * 1.1)  # Set y-axis limit for better visualization
    plt.show()


#------ data for plotting Cv vs T for different x_e values ------
T_range = np.logspace(-1, 2, 400)  # Temperature range
x_e_list = np.array([0.000001, 0.0001, 0.001, 0.01, 0.05])
x_e = 0.05

#plot_Cv_Morse(T_range, x_e)  # Run plot function for each x_e value
plot_Cv_Morse_Xe_list(T_range, x_e_list)  # Run plot function for a list of x_e values