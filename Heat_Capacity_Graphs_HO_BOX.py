import numpy as np
import matplotlib.pyplot as plt

# Constants (normalized so kB = 1 for simplicity)
kB = 1.0
m = 1.0
L = 1.0
h_bar = 0.1
E_g = (h_bar**2 * np.pi**2) / (2 * m * L**2)  # Ground state energy for box potential
tol = 1e-4
iter = 1e4

#Number of Box Potential Energy Levels
xi_initial = 5
n_levels = [100, 500]
n = 1e5

# Temperature range (log scale)
T_span = np.logspace(-2, 2, 400)

# --- Harmonic Oscillator Heat Capacity ---

def Cv_HO_Quantum(T, hw=1.0):
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

def Cv_Box_Classical(T, n, xi): # retuns Cv with a scaling factor for a specific T, n, and xi
    # temp_ratio = kB * T / Eg
    # En = n^2 * Eg
    n_array = np.arange(1, n + 1)
    
    E_n = n_array**2 * E_g / xi**2
    beta = 1.0 / (T * kB * xi**2)
    beta_exp = 1.0 / (T * kB)

    exp_shift = - beta_exp * E_n[0]  # Shift to prevent underflow
    exp = np.exp(-beta_exp * E_n - exp_shift)
    
    # Partition function and its derivatives
    z = np.sum(exp)

    # Catch precision collapse when states become completely indistinguishable
    if z == 0 or np.all(exp == exp[0]):
        return 0.0
    
    E_ave = np.sum(E_n * exp) / z
    E2_ave = np.sum(E_n**2 * exp) / z
    
    variance = E2_ave - E_ave**2
    if variance <= 0:
        return 0.0
    
    # Cv = ( <E^2> - <E>^2 ) / (kB * T^2)
    Cv = variance * beta**2 * kB

    return Cv

def check_n_convergance(T, n, xi, tolerance=tol, iterations=iter):
    converged = False
    counter = 0

    n_curr = n
    Cv_curr = Cv_Box_Classical(T, n_curr, xi)
    
    while not converged and counter < iterations:
        n_next = n_curr + 1
        Cv_next = Cv_Box_Classical(T, n_next, xi)
        
        rmse = np.sqrt(np.mean((Cv_next - Cv_curr)**2))
        
        if rmse < tolerance:
            converged = True
        else:
            n_curr = n_next
            Cv_curr = Cv_next
        
        counter += 1
    if not converged:
        raise ValueError(f"Warning: tolerance ({tolerance}) is too small. N did not converge.")
    elif counter >= iterations:
        raise ValueError(f"Warning: n did not converge within the maximum number of iterations ({iterations}).")
    else:
        return n_curr
    
def check_xi_convergence(T, n, xi, tolerance=tol, iterations=iter):
    converged = False
    counter = 0

    xi_curr = xi
    Cv_curr = Cv_Box_Classical(T, n, xi_curr)
    
    while not converged and counter < iterations:
        xi_next = xi_curr + 1
        Cv_next = Cv_Box_Classical(T, n, xi_next)
        
        rmse = np.sqrt(np.mean((Cv_next - Cv_curr)**2))
        
        if rmse < tolerance:
            converged = True
        else:
            xi_curr = xi_next
            Cv_curr = Cv_next
        
        counter += 1
    if not converged:
        raise ValueError(f"Warning: tolerance ({tolerance}) is too small. Xi did not converge to classical limit.")
    elif counter >= iterations:
        raise ValueError(f"Warning: xi did not converge within the maximum number of iterations ({iterations}).")
    else:
        return xi_curr

def find_classical_limit_Box(T_span, n_initial, xi_initial, tolerance=tol):
    Cv_classical = []

    for T in T_span:
        n = check_n_convergance(T, n_initial, xi_initial)
        xi = check_xi_convergence(T, n, xi_initial)
        classical_limit = Cv_Box_Classical(T, n, xi)
        Cv_classical.append(classical_limit)
    return np.array(Cv_classical)


    
        
        

# Create the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Left Plot: Harmonic Potential ---
ax1.plot(T_span, [kB]*len(T_span), '--', color='yellowgreen', label='Classical')
ax1.plot(T_span, Cv_HO_Quantum(T_span), '-', color='steelblue', label='Quantum')
ax1.set_title("Harmonic potential")
ax1.set_xlabel(r"$k_B T / E_g$")
ax1.set_ylabel(r"$C_v [J/K]$")
ax1.set_xscale('log')
ax1.set_ylim(0, 1.2)
ax1.legend()

# --- Right Plot: Box Potential ---
classical_limit_box = find_classical_limit_Box(T_span, n, xi_initial)
print(f"Classical limit for box potential converged at xi = {check_xi_convergence(T_span[0], n, xi_initial)}")
ax2.plot(T_span, classical_limit_box, '--', 
        color='yellowgreen', label=f'Classical Limit')
ax2.plot(T_span, Cv_Box_Quantum(T_span, n), '-', label=f'Quantum (n={n})')
ax2.set_title("Box potential")
ax2.set_xlabel(r"$k_B T / E_g$")
ax2.set_ylabel(r"$C_v [J/K]$")
ax2.set_xscale('log')
ax2.set_ylim(0, 0.7)
ax2.legend()

plt.tight_layout()
plt.show()


# For multiple n_levels in the box potential plot
'''
ax2.plot(T_span, [0.5*kB]*len(T_span), '--', color='yellowgreen', label='Classical')
for n in n_levels:
    ax2.plot(T_span, Cv_Box_Quantum(T_span, n), '-', label=f'Quantum (n={n})')
ax2.set_title("Box potential")
ax2.set_xlabel(r"$k_B T / E_g$")
ax2.set_ylabel(r"$C_v [J/K]$")
ax2.set_xscale('log')
ax2.set_ylim(0, 0.7)
ax2.legend()
'''

# Create Classical limit plot for the box potential
'''
fig = plt.figure(figsize=(6, 5))
classical_limit_box, xi_converged = find_classical_limit_Box(T_span, n)
plt.plot(T_span, classical_limit_box, '--', color='yellowgreen', label=f'Classical Limit (xi={xi_converged:.2f})')
plt.title("Classical Limit for Box Potential")
plt.xlabel(r"$k_B T / E_g$")
plt.ylabel(r"$C_v [J/K]$")
plt.xscale('log')
plt.ylim(0, 0.7)
plt.legend()
plt.tight_layout()
plt.show()
'''
