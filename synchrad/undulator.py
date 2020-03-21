import numpy as np
import scipy.integrate
import scipy.special

c_light = 299792458
elementary_charge = 1.60217662e-19
vacuum_permittivity = 8.8541878128e-12
electron_mass = 9.1093837015e-31
hbar = 1.054571817e-34
hbar_ev = 6.582119569e-16 # eV * s
joules_per_ev = elementary_charge

def sinc(x):
    return np.sin(x) / x

def weak_undulator_trajectory(K, k_u, gamma, time_step, steps):
    beta = np.sqrt(1 - gamma ** -2)
    a = K / (beta * gamma * k_u)
    t = time_step * np.arange(steps + 1)
    omega_u = beta * c_light * k_u
    trajectory = np.empty(dtype=np.float64, shape=(1, steps + 1, 9))
    trajectory[0, :, 0] = a * np.cos(omega_u * t)
    trajectory[0, :, 1] = 0
    trajectory[0, :, 2] = (beta - 1) * c_light * t
    trajectory[0, :, 3] = (-a * omega_u / c_light) * np.sin(omega_u * t)
    trajectory[0, :, 4] = 0
    trajectory[0, :, 5] = gamma
    trajectory[0, :, 6] = (-a * omega_u ** 2 / c_light) * np.cos(omega_u * t)
    trajectory[0, :, 7] = 0
    trajectory[0, :, 8] = 0
    return trajectory

def strong_undulator_trajectory(K, k_u, gamma, time_step, steps):
    beta = np.sqrt(1 - gamma ** -2)
    a = K / (beta * gamma * k_u)
    t = time_step * np.arange(steps + 1)
    beta_star = beta * (1 - (K / (2 * beta * gamma)) ** 2)
    omega_u = beta_star * c_light * k_u
    trajectory = np.empty(dtype=np.float64, shape=(1, steps + 1, 9))
    trajectory[0, :, 0] = a * np.cos(omega_u * beta_star * t)
    trajectory[0, :, 1] = 0
    trajectory[0, :, 2] = (beta_star - 1) * c_light * t + (K ** 2 / (8 * beta ** 2 * gamma ** 2 * k_u)) * np.sin(2 * omega_u * t)
    trajectory[0, :, 3] = (-a * omega_u / c_light) * np.sin(omega_u * t)
    trajectory[0, :, 4] = 0
    trajectory[0, :, 5] = gamma
    trajectory[0, :, 6] = (-a * omega_u ** 2 / c_light) * np.cos(omega_u * t)
    trajectory[0, :, 7] = 0
    trajectory[0, :, 8] = 0
    return trajectory

def weak_undulator_double_differential(energies, phi_xs, phi_ys, K, k_u, N_u, gamma):
    new_energies = energies[:, np.newaxis, np.newaxis]
    new_thetas = np.sqrt(phi_xs[np.newaxis, :, np.newaxis] ** 2 + phi_ys[np.newaxis, np.newaxis, :] ** 2)
    new_phis = np.arctan2(phi_ys[np.newaxis, np.newaxis, :], phi_xs[np.newaxis, :, np.newaxis])
    beta = np.sqrt(1 - gamma ** -2)
    omega_u = beta * c_light * k_u
    a = K / (beta * gamma * k_u)
    constant_factor = elementary_charge ** 2 * k_u * K ** 2 * gamma ** 4 * N_u ** 2 / (2 * np.pi * vacuum_permittivity * beta * joules_per_ev)
    gamma_theta_2 = (gamma * new_thetas) ** 2
    energy_1 = 2 * hbar_ev * gamma ** 2 * omega_u / (1 + gamma_theta_2)
    sinc_factor = sinc(np.pi * N_u * (new_energies / energy_1 - 1)) ** 2
    angle_factor = (1 - 2 * gamma_theta_2 * np.cos(2 * new_phis) + gamma_theta_2 ** 2) / ((1 + gamma_theta_2) ** 5)
    return constant_factor * angle_factor * sinc_factor / energy_1

def strong_undulator_double_differential(energies, phi_xs, phi_ys, K, k_u, N_u, gamma, N, M):
    # resize input arrays
    new_energies = energies[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    new_thetas = np.sqrt(phi_xs[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis] ** 2 + phi_ys[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] ** 2)
    new_phis = np.arctan2(phi_ys[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis], phi_xs[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis])
    # constants
    beta = np.sqrt(1 - gamma ** -2)
    a = K / (beta * gamma * k_u)
    beta_star = beta * (1 - (K / (2 * beta * gamma)) ** 2)
    omega_u = beta_star * c_light * k_u
    gamma_star = 1 / np.sqrt(1 - beta_star ** 2)
    K_star = K / np.sqrt(1 + 0.5 * K ** 2)
    constant_factor = N_u * gamma_star ** 2 * gamma ** 2 * elementary_charge ** 2 * k_u * K ** 2 / (6 * vacuum_permittivity * hbar)
    # compute some values
    gamma_star_theta_2 = (gamma_star * new_thetas) ** 2
    a_u = K_star ** 2 / (4 * (1 + gamma_star_theta_2))
    b_u = 2 * K_star * gamma_star * new_thetas * np.cos(new_phis) / (1 + gamma_star_theta_2)
    m = np.arange(1, N + 1)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    l = np.arange(-M, M + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    jvlmau = scipy.special.jv(l, m * a_u)
    sigma_m_1 = np.sum(jvlmau * scipy.special.jv(m + 2 * l, m * b_u), axis=4)
    sigma_m_2 = np.sum(jvlmau * (scipy.special.jv(m + 2 * l + 1, m * b_u) + scipy.special.jv(m + 2 * l - 1, m * b_u)), axis=4)
    # reshape things
    new_energies = new_energies.reshape(new_energies.shape[:-1])
    new_thetas = new_thetas.reshape(new_thetas.shape[:-1])
    new_phis = new_phis.reshape(new_phis.shape[:-1])
    gamma_star_theta_2 = gamma_star_theta_2.reshape(gamma_star_theta_2.shape[:-1])
    m = m.reshape(m.shape[:-1])
    # compute some more values
    angle_factor = (3 * m ** 2 / (np.pi * (1 + 0.5 * K ** 2) ** 2 * K_star ** 2)) * \
        (4 * sigma_m_1 ** 2 * gamma_star_theta_2 - 4 * gamma_star * new_thetas * sigma_m_1 * sigma_m_2 * K_star * np.cos(new_phis) + (sigma_m_2 * K_star) ** 2) \
        / ((1 + gamma_star_theta_2) ** 3)
    omega = new_energies / hbar_ev
    omega_1 = 2 * gamma_star ** 2 * omega_u / (1 + gamma_star_theta_2)
    sinc_factor = (N_u / omega_1) * sinc(np.pi * N_u * (omega / omega_1 - m)) ** 2
    return constant_factor * np.sum(angle_factor * sinc_factor, axis=3)
