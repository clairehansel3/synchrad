from .core import track_beam_linear, compute_radiation_grid, linspace_midpoint, logspace_midpoint
from .plotting import plot_spot, plot_double_differential
import numpy as np
import scipy.integrate
import scipy.interpolate

c_light = 299792458

def get_function(particles, gamma_initial, ion_atomic_number, plasma_density,
        accelerating_field, spot_size, normalized_emittance, plasma_length,
        steps, energy_exponent_min, energy_exponent_max, energy_points,
        theta_min, theta_max, theta_points, threads=1, filename=None,
        filename_dd=None):
    time_step = plasma_length / (steps * c_light)
    trajectories = track_beam_linear(particles, gamma_initial,
        ion_atomic_number, plasma_density, accelerating_field, spot_size,
        normalized_emittance, time_step, steps)
    energies, energies_midpoint = logspace_midpoint(energy_exponent_min,
        energy_exponent_max, energy_points)
    thetas, thetas_midpoint, thetas_step = linspace_midpoint(theta_min,
        theta_max, theta_points)
    print('-> begin')
    result = compute_radiation_grid(trajectories, energies, thetas,
        np.array([0.0]), time_step, threads=threads)
    print('-> end')
    double_differential = np.sum(result ** 2, axis=3)
    if filename_dd is not None:
        plot_double_differential(double_differential, energies,
            energies_midpoint, thetas, thetas_midpoint, filename_dd)
    spectrum = 2 * np.pi * np.sum(double_differential[:, :, 0]
        * thetas[np.newaxis, :], axis=1) * thetas_step
    f = scipy.interpolate.interp1d(energies, spectrum)
    normalization_factor, _ = scipy.integrate.quad(f, 10 ** energy_exponent_min,
        10 ** energy_exponent_max)
    print('norm: ', normalization_factor, _)
    if filename is not None:
        plot_spot(trajectories, time_step, filename, spot_size=spot_size)
    return scipy.interpolate.interp1d(energies, spectrum / normalization_factor)

def get_function_and_error(particles, gamma_initial, ion_atomic_number,
        plasma_density, accelerating_field, spot_size, normalized_emittance,
        plasma_length, steps, energy_exponent_min, energy_exponent_max,
        energy_points, theta_min, theta_max, theta_points, threads=1,
        evaluations=5):
    functions = []
    for _ in range(evaluations):
        functions.append(get_function(particles, gamma_initial,
            ion_atomic_number, plasma_density, accelerating_field, spot_size,
            normalized_emittance, plasma_length, steps, energy_exponent_min,
            energy_exponent_max, energy_points, theta_min, theta_max,
            theta_points, threads=threads))
    def f(energy):
        results = np.empty((evaluations, len(energy)))
        for i, function in enumerate(functions):
            results[i, :] = function(energy)
        return np.mean(results, axis=0), np.std(results, axis=0)
    return f, functions

def sample_from_spectrum(f, energy_exponent_min, energy_exponent_max):
    energies = np.linspace(10 ** energy_exponent_min, 10 ** energy_exponent_max,
        10000)
    max = np.max(f(energies))
