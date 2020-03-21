import ctypes
import numpy as np
import pathlib
import random

synchrad_cxx_library = ctypes.cdll.LoadLibrary(pathlib.Path(__file__).parent.parent / 'libsynchrad.so')

c_light = 299792458
classical_electron_radius = 2.8179403227e-15

def linspace_midpoint(min, max, num):
    vals, step = np.linspace(min, max, num, endpoint=True, retstep=True)
    vals_midpoint = np.linspace((min - 0.5 * step), (max + 0.5 * step), (num + 1), endpoint=True)
    return (vals, vals_midpoint, step)

def logspace_midpoint(min, max, num):
    vals, step = np.linspace(min, max, num, endpoint=True, retstep=True)
    vals_midpoint = np.linspace((min - 0.5 * step), (max + 0.5 * step), (num + 1), endpoint=True)
    return 10 ** vals, 10 ** vals_midpoint

def raise_c_contiguous_error_msg(parameter_name, function_name):
    raise ValueError(('The parameter \'{parameter_name}\' to the function \'{f'\
        'unction_name}\' is not c contiguous. Try passing a copy of \'{paramet'\
        'er_name}\' instead. e.g. {function_name}(..., {parameter_name}.copy()'\
        ', ...)}').format(parameter_name=parameter_name,
        function_name=function_name))

def track_particle_ion_collapse(x0, y0, vx0, vy0, gamma_initial, ion_atomic_number,
        plasma_density, rho_ion, accelerating_field, bennett_radius_initial,
        time_step, steps):
    """
    Tracks a single electron in the ion collapse case.

    inputs:
        x0: initial x [m]
        y0: initial y [m]
        vx0: initial vx [m/s]
        vy0: initial vy [m/s]
        gamma_initial: gamma at t = 0
        ion_atomic_number: Z
        plasma_density: background plasma ion density [m^-3]
        rho_ion: rho_ion from bennett profile [m^-3]
        accelerating_field: constant z accelerating field, note that a negative
            value accelerates the electron while a positive value decellerates
            it [V/m]
        bennett_radius_initial: bennett radius at t = 0 [m]
        time_step: step size [s]
        steps: number of time steps

    returns:
        numpy array of np.float64 with shape (1, steps + 1, 9). The first index
        is the particle, the second index is the step, and the third index is
        the coordinate. The coordinates are
        0 x [m]
        1 y [m]
        2 z - ct [m]
        3 beta_x
        4 beta_y
        5 gamma
        6 beta_x_dot [1/s]
        7 beta_y_dot [1/s]
        8 gamma_dot [1/s]
    """
    result = np.empty(dtype=np.float64, shape=(1, steps + 1, 9))
    synchrad_cxx_library.track_particle(
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(x0),
        ctypes.c_double(y0),
        ctypes.c_double(vx0),
        ctypes.c_double(vy0),
        ctypes.c_double(gamma_initial),
        ctypes.c_double(ion_atomic_number),
        ctypes.c_double(plasma_density),
        ctypes.c_double(rho_ion),
        ctypes.c_double(accelerating_field),
        ctypes.c_double(bennett_radius_initial),
        ctypes.c_double(time_step),
        ctypes.c_size_t(steps)
    )
    return result

def track_beam_ion_collapse(particles, gamma_initial, ion_atomic_number,
        plasma_density, rho_ion, accelerating_field, bennett_radius_initial,
        time_step, steps, seed=None):
    """
    Tracks a beam of electrons in the ion collapse case. Electrons are randomly
    sampled from the equilibrium distribution.

    inputs:
        particles: number of particles to track
        gamma_initial: gamma at t = 0
        ion_atomic_number: Z
        plasma_density: background plasma ion density [m^-3]
        rho_ion: rho_ion from bennett profile [m^-3]
        accelerating_field: constant z accelerating field, note that a negative
            value accelerates the electron while a positive value decellerates
            it [V/m]
        bennett_radius_initial: bennett radius at t = 0 [m]
        time_step: step size [s]
        steps: number of time steps

    returns:
        numpy array of np.float64 with shape (particles, steps + 1, 9). The
        first index is the particle, the second index is the step, and the third
        index is the coordinate. The coordinates are
        0 x [m]
        1 y [m]
        2 z - ct [m]
        3 beta_x
        4 beta_y
        5 gamma
        6 beta_x_dot [1/s]
        7 beta_y_dot [1/s]
        8 gamma_dot [1/s]
    """
    sigma_r = 0.5 * bennett_radius_initial* np.sqrt(rho_ion / plasma_density)
    sigma_r_dot = bennett_radius_initial * c_light * np.sqrt(np.pi * classical_electron_radius * ion_atomic_number * rho_ion / (2 * gamma_initial))
    random.seed(a=seed)
    result = np.empty(dtype=np.float64, shape=(particles, steps + 1, 9))
    for particle in range(particles):
        while True:
            test_r = bennett_radius_initial / np.sqrt(1 / random.random() - 1)
            if (random.random() < np.exp(-test_r ** 2 / (2 * sigma_r ** 2))):
                r = test_r
                break
        theta = 2 * np.pi * random.random()
        r_dot = random.normalvariate(0.0, sigma_r_dot)
        r_theta_dot = random.normalvariate(0.0, sigma_r_dot)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        vx = (r_dot * np.cos(theta) - r_theta_dot * np.sin(theta))
        vy = (r_dot * np.sin(theta) + r_theta_dot * np.cos(theta))
        result[particle, :, :] = track_particle_ion_collapse(x, y, vx, vy,
            gamma_initial, ion_atomic_number, plasma_density, rho_ion,
            accelerating_field, bennett_radius_initial, time_step, steps)
    return result

def compute_radiation(trajectories, energies, phi_xs, phi_ys, time_step):
    """
    Computes d2U / (dOmega depsilon) from a set of trajectories. Scans over all
    combinations of photon energies and angles.

    inputs:
        trajectories: a numpy array of np.float64 with shape
            (particles, steps + 1, 9). The first index is the particle, the
            second index is the step, and the third index is the coordinate. The
            coordinates are
            0 x [m]
            1 y [m]
            2 z - ct [m]
            3 beta_x
            4 beta_y
            5 gamma
            6 beta_x_dot [1/s]
            7 beta_y_dot [1/s]
            8 gamma_dot [1/s]
        energies: a 1d numpy array of np.float64 containing values of the photon
            energies in eV. [eV]
        phi_xs: a 1d numpy array of np.float64 containing projected angles in x
            [rad]
        phi_ys: a 1d numpy array of np.float64 containing projected angles in y
            [rad]
        time_step: step size [s]

    returns:
        numpy array of np.float64 with shape
        (n_energies, n_phi_xs, n_phi_ys, 6). result[i, j, k] is the value of the
        vector V for energy energies[i] and angles phi_xs[j] and phi_ys[k]. The
        vector V is a 3d complex vector: V[0] = re(V_x), V[1] = im(V_x),
        V[2] = re(V_y), etc. The squared norm of V is np.sum(V ** 2, axis=3) is
        d2U / (dOmega depsilon). If V1 and V2 are the radiation from two sets of
        particles P1 and P2, V1 + V2 is the radiation from all the particles in
        P1 and P2.
    """
    assert all(isinstance(arg, np.ndarray) for arg in (trajectories, energies, phi_xs, phi_ys))
    assert all(arg.dtype == np.float64 for arg in (trajectories, energies, phi_xs, phi_ys))
    assert trajectories.ndim == 3
    assert trajectories.shape[2] == 9
    assert energies.ndim == phi_xs.ndim == phi_ys.ndim == 1
    if not trajectories.flags['C_CONTIGUOUS']:
        raise_c_contiguous_error_msg('trajectories', 'compute_radiation')
    if not energies.flags['C_CONTIGUOUS']:
        raise_c_contiguous_error_msg('energies', 'compute_radiation')
    if not phi_xs.flags['C_CONTIGUOUS']:
        raise_c_contiguous_error_msg('phi_xs', 'compute_radiation')
    if not phi_ys.flags['C_CONTIGUOUS']:
        raise_c_contiguous_error_msg('phi_ys', 'compute_radiation')
    n_particles = trajectories.shape[0]
    n_steps = trajectories.shape[1] - 1
    n_energies = energies.shape[0]
    n_phi_xs = phi_xs.shape[0]
    n_phi_ys = phi_ys.shape[0]
    result = np.empty(dtype=np.float64, shape=(n_energies, n_phi_xs, n_phi_ys, 6))
    synchrad_cxx_library.compute_radiation(
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        trajectories.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        phi_xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        phi_ys.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(n_particles),
        ctypes.c_size_t(n_steps),
        ctypes.c_size_t(n_energies),
        ctypes.c_size_t(n_phi_xs),
        ctypes.c_size_t(n_phi_ys),
        ctypes.c_double(time_step)
    )
    return result

def compute_radiation2(trajectories, inputs, time_step):
    """
    Computes d2U / (dOmega depsilon) from a set of trajectories for different
    photon energies and angles.

    inputs:
        trajectories: a numpy array of np.float64 with shape
            (particles, steps + 1, 9). The first index is the particle, the
            second index is the step, and the third index is the coordinate. The
            coordinates are
            0 x [m]
            1 y [m]
            2 z - ct [m]
            3 beta_x
            4 beta_y
            5 gamma
            6 beta_x_dot [1/s]
            7 beta_y_dot [1/s]
            8 gamma_dot [1/s]
        inputs: a numpy array of np.float64 with shape (n_inputs, 3). This
            function evaluates n_inputs different values of the radiation. For
            the ith value where 0 <= i < n_inputs, the photon energy is
            inputs[i, 0] (units: eV), phi_x is inputs[i, 1] (units: rad), and
            phi_y is inputs[i, 2] (units: rad).
        time_step: step size [s]

    returns:
        numpy array of np.float64 with shape
        (n_inputs, 6). result[i] is the value of the vector V for the ith input.
        The vector V is a 3d complex vector: V[0] = re(V_x), V[1] = im(V_x),
        V[2] = re(V_y), etc. The squared norm of V is np.sum(V ** 2, axis=3) is
        d2U / (dOmega depsilon). If V1 and V2 are the radiation from two sets of
        particles P1 and P2, V1 + V2 is the radiation from all the particles in
        P1 and P2.
    """
    assert all(isinstance(arg, np.ndarray) for arg in (trajectories, inputs))
    assert all(arg.dtype == np.float64 for arg in (trajectories, inputs))
    assert trajectories.ndim == 3
    assert trajectories.shape[2] == 9
    assert inputs.ndim == 2
    assert inputs.shape[1] == 3
    if not trajectories.flags['C_CONTIGUOUS']:
        raise_c_contiguous_error_msg('trajectories', 'compute_radiation2')
    if not inputs.flags['C_CONTIGUOUS']:
        raise_c_contiguous_error_msg('inputs', 'compute_radiation2')
    n_particles = trajectories.shape[0]
    n_steps = trajectories.shape[1] - 1
    n_inputs = inputs.shape[0]
    result = np.empty(dtype=np.float64, shape=(n_inputs, 6))
    synchrad_cxx_library.compute_radiation2(
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        trajectories.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(n_particles),
        ctypes.c_size_t(n_steps),
        ctypes.c_size_t(n_inputs),
        ctypes.c_double(time_step)
    )
    return result
