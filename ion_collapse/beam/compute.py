import synchrad
import numpy as np
import sys

particles = 10
bennett_radius_initial = 170e-9
ion_atomic_number = 1
gamma_initial = 10000 / 0.510998
plasma_density = 1e24
rho_ion = 1e26
steps = 200000
accelerating_field = -50e9
plasma_length = 0.1
time_step = plasma_length / (steps * 299792458)
energies, energies_midpoint = synchrad.logspace_midpoint(5, 8, 200)
thetas, thetas_midpoint, thetas_step = synchrad.linspace_midpoint(0, 3e-3, 200)

if __name__ == '__main__':
    trajectory = synchrad.track_beam_ion_collapse(particles, gamma_initial,
        ion_atomic_number, plasma_density, rho_ion, accelerating_field,
        bennett_radius_initial, time_step, steps)
    trajectory.tofile(f'data/trajectory_{sys.argv[1]}')
    result = synchrad.compute_radiation(trajectory, energies, thetas, np.array([0.0]), time_step)
    result.tofile(f'data/radiation_{sys.argv[1]}')
