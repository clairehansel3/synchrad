import synchrad
import numpy as np
import sys

i = int(sys.argv[1])

bennett_radius_initial = 170e-9
ion_atomic_number = 1
gamma_initial = 10000 / 0.510998
plasma_density = 1e24
rho_ion = 1e26
steps = 20000
accelerating_field = 0
plasma_length = 0.01
time_step = plasma_length / (steps * 299792458)
x0_values = bennett_radius_initial * np.linspace(0, 5, 1000)

energies, energies_midpoint = synchrad.logspace_midpoint(5, 8, 101)
phi_xs, phi_xs_midpoint, phi_xs_step = synchrad.linspace_midpoint(-4e-3, 4e-3, 101)
phi_ys, phi_ys_midpoint, phi_ys_step = synchrad.linspace_midpoint(-10e-5, 10e-5, 101)
zero_index = 50

if __name__ == '__main__':
    trajectory = synchrad.track_particle_ion_collapse(x0_values[i], 0, 0, 0,
        gamma_initial, ion_atomic_number, plasma_density, rho_ion,
        accelerating_field, bennett_radius_initial, time_step, steps)
    trajectory.tofile(f'data/trajectory_{i}')
    result = synchrad.compute_radiation(trajectory, energies, phi_xs, phi_ys, time_step)
    double_differential = np.sum(result ** 2, axis=3)
    double_differential.tofile(f'data/double_differential_{i}')
