import synchrad
import synchrad.plotting
import synchrad.undulator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import scipy.integrate

electron_rest_energy_mev = 0.510998
c_light = 299792458
classical_electron_radius = 2.8179403227e-15

def validate_beam_tracking():
    particles = 10000
    plasma_length = 0.1
    bennett_radius_initial = 170e-9
    time_step = 10 ** -5 / c_light
    gamma_initial = 10 * (1000 / electron_rest_energy_mev)
    ion_atomic_number = 1
    plasma_density = 1e24
    rho_ion = 1e26
    accelerating_field = -100e9
    steps = int(round(plasma_length / (c_light * time_step)))
    trajectories = synchrad.track_beam_ion_collapse(particles, gamma_initial,
        ion_atomic_number, plasma_density, rho_ion, accelerating_field,
        bennett_radius_initial, time_step, steps)
    t = time_step * np.arange(steps + 1)
    # r
    gamma_final = gamma_initial + (-plasma_length * accelerating_field / (electron_rest_energy_mev * 1e6))
    bennett_radius_final = bennett_radius_initial * (gamma_final / gamma_initial) ** -0.25
    sigma_r = 0.5 * bennett_radius_final * np.sqrt(rho_ion / plasma_density)
    sigma_r_dot = bennett_radius_final * c_light * np.sqrt(np.pi * classical_electron_radius * ion_atomic_number * rho_ion / (2 * gamma_final))
    x = trajectories[:, -1, 0]
    y = trajectories[:, -1, 1]
    vx = trajectories[:, -1, 3] * c_light
    vy = trajectories[:, -1, 4] * c_light
    r = np.sqrt(x ** 2 + y ** 2)
    th = np.arctan2(y, x)
    vr = (x * vx + y * vy) / r
    rvth = (x * vy - y * vx) / r
    r = r[r < 5 * bennett_radius_final]
    vr = vr[np.logical_and(-5 * sigma_r_dot <= vr, vr <= 5 * sigma_r_dot)]
    rvth = rvth[np.logical_and(-5 * sigma_r_dot <= rvth, rvth <= 5 * sigma_r_dot)]
    unnormalized_pdf = lambda r: r * np.exp(-r ** 2 / (2 * sigma_r ** 2)) / ((1 + ((r / bennett_radius_final) ** 2)) ** 2)
    normalization_factor = scipy.integrate.quad(unnormalized_pdf, 0, 5 * bennett_radius_final)[0]
    r_pdf = lambda r: unnormalized_pdf(r) / normalization_factor

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(6.4, 4.8))

    fig.suptitle('Beam Distribution at End of Plasma')

    ax1.hist(r, density=True, bins=100, range=(0, 5 * bennett_radius_final))
    ax1.plot(np.linspace(0, 5 * bennett_radius_final, 1000), r_pdf(np.linspace(0, 5 * bennett_radius_final, 1000)))
    ax1.set_xlabel(r'$r$')
    ax1.set_xticks([0, 5 * bennett_radius_final])
    ax1.set_xticklabels([r'$0$', r'$5a$'])
    ax1.set_yticks([])
    ax1.set_yticklabels([])

    ax2.hist(th, density=True, bins=100, range=(-np.pi, np.pi))
    ax2.plot([-np.pi, np.pi], [1 / (2 * np.pi), 1 / (2 * np.pi)])
    ax2.set_xlabel(r'$\theta$')
    ax2.set_xticks([-np.pi, 0, np.pi])
    ax2.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax2.set_yticks([])
    ax2.set_yticklabels([])

    v_values = np.linspace(-5 * sigma_r_dot, 5 * sigma_r_dot, 1000)

    ax3.hist(vr, density=True, bins=100, range=(-5 * sigma_r_dot, 5 * sigma_r_dot))
    ax3.plot(v_values, np.exp(-v_values ** 2 / (2 * sigma_r_dot ** 2)) / (np.sqrt(2 * np.pi) * sigma_r_dot))
    ax3.set_xlabel(r'$\dot{r}$')
    ax3.set_xticks([-5*sigma_r_dot, 0, 5*sigma_r_dot])
    ax3.set_xticklabels([r'$-5\sigma_{\dot{r}}$', r'$0$', r'5$\sigma_{\dot{r}}$'])
    ax3.set_yticks([])
    ax3.set_yticklabels([])

    ax4.hist(rvth, density=True, bins=100, range=(-5 * sigma_r_dot, 5 * sigma_r_dot))
    ax4.plot(v_values, np.exp(-v_values ** 2 / (2 * sigma_r_dot ** 2)) / (np.sqrt(2 * np.pi) * sigma_r_dot))
    ax4.set_xlabel(r'$r \dot{\theta}$')
    ax4.set_xticks([-5*sigma_r_dot, 0, 5*sigma_r_dot])
    ax4.set_xticklabels([r'$-5 \sigma_{r\dot{\theta}}$', r'$0$', r'5 $\sigma_{r\dot{\theta}}$'])
    ax4.set_yticks([])
    ax4.set_yticklabels([])

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig('equilibrium.png', dpi=300)
    plt.close(fig)


def validate_weak_undulator():
    K = 0.01
    lambda_u = 0.01
    N_u = 10
    energy_mev = 100
    steps = 1000

    time_step = N_u * lambda_u / (steps * c_light)
    k_u = 2 * np.pi / lambda_u
    gamma = energy_mev / electron_rest_energy_mev
    trajectory = synchrad.undulator.weak_undulator_trajectory(K, k_u, gamma, time_step, steps)
    energies, energies_midpoint, energies_step = synchrad.linspace_midpoint(1, 12, 200)
    phi_xs, phi_xs_midpoint, phi_xs_step = synchrad.linspace_midpoint(0, 5e-3, 200)
    result_numerical = synchrad.compute_radiation(trajectory, energies, phi_xs, np.array([0.0]), time_step)
    dd_numerical = np.sum(result_numerical ** 2, axis=3)
    dd_analytic = synchrad.undulator.weak_undulator_double_differential(energies, phi_xs, np.array([0.0]), K, k_u, N_u, gamma)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(plt.rcParams['figure.figsize'][0] * 1.5, plt.rcParams['figure.figsize'][1]))

    vmin1, vmax1 = dd_analytic.min(), dd_analytic.max()
    ax1.set_title('analytic')
    ax1.pcolormesh(energies_midpoint, phi_xs_midpoint, dd_analytic[:, :, 0].T, vmin=vmin1, vmax=vmax1, cmap='inferno')
    ax1.set_xlim(energies.min(), energies.max())
    ax1.set_ylim(phi_xs.min(), phi_xs.max())
    ax1.set_xlabel('photon energy (eV)')
    ax1.set_ylabel('$\\theta$ (rad)')
    ax1.yaxis.get_major_formatter().set_powerlimits((0, 1))
    cbar1 = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin1, vmax=vmax1), cmap='inferno'), ax=ax1)
    cbar1.set_label('$\\frac{d^2 U}{d\\Omega d\\epsilon}$')
    cbar1.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    vmin2, vmax2 = dd_numerical.min(), dd_numerical.max()
    ax2.set_title('numerical')
    ax2.pcolormesh(energies_midpoint, phi_xs_midpoint, dd_numerical[:, :, 0].T, vmin=vmin2, vmax=vmax2, cmap='inferno')
    ax2.set_xlim(energies.min(), energies.max())
    ax2.set_ylim(phi_xs.min(), phi_xs.max())
    ax2.set_xlabel('photon energy (eV)')
    ax2.yaxis.get_major_formatter().set_powerlimits((0, 1))
    cbar2 = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin2, vmax=vmax2), cmap='inferno'), ax=ax2)
    cbar2.set_label('$\\frac{d^2 U}{d\\Omega d\\epsilon}$')
    cbar2.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    fig.savefig('weak_undulator.png', dpi=300)
    plt.close(fig)

def validate_strong_undulator():
    K = 2
    lambda_u = 0.01
    N_u = 10
    energy_mev = 100
    steps = 1000

    time_step = N_u * lambda_u / (steps * c_light)
    k_u = 2 * np.pi / lambda_u
    gamma = energy_mev / electron_rest_energy_mev
    trajectory = synchrad.undulator.strong_undulator_trajectory(K, k_u, gamma, time_step, steps)
    energies, energies_midpoint, energies_step = synchrad.linspace_midpoint(1, 100, 200)
    phi_xs, phi_xs_midpoint, phi_xs_step = synchrad.linspace_midpoint(0, 2e-2, 200)
    result_numerical = synchrad.compute_radiation(trajectory, energies, phi_xs, np.array([0.0]), time_step)
    dd_numerical = np.sum(result_numerical ** 2, axis=3)
    dd_analytic = synchrad.undulator.strong_undulator_double_differential(energies, phi_xs, np.array([0.0]), K, k_u, N_u, gamma, 100, 100)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(plt.rcParams['figure.figsize'][0] * 1.5, plt.rcParams['figure.figsize'][1]))

    vmin1, vmax1 = dd_analytic.min(), dd_analytic.max()
    ax1.set_title('analytic')
    ax1.pcolormesh(energies_midpoint, phi_xs_midpoint, dd_analytic[:, :, 0].T, vmin=vmin1, vmax=vmax1, cmap='inferno')
    ax1.set_xlim(energies.min(), energies.max())
    ax1.set_ylim(phi_xs.min(), phi_xs.max())
    ax1.set_xlabel('photon energy (eV)')
    ax1.set_ylabel('$\\theta$ (rad)')
    ax1.yaxis.get_major_formatter().set_powerlimits((0, 1))
    cbar1 = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin1, vmax=vmax1), cmap='inferno'), ax=ax1)
    cbar1.set_label('$\\frac{d^2 U}{d\\Omega d\\epsilon}$')
    cbar1.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    vmin2, vmax2 = dd_numerical.min(), dd_numerical.max()
    ax2.set_title('numerical')
    ax2.pcolormesh(energies_midpoint, phi_xs_midpoint, dd_numerical[:, :, 0].T, vmin=vmin2, vmax=vmax2, cmap='inferno')
    ax2.set_xlim(energies.min(), energies.max())
    ax2.set_ylim(phi_xs.min(), phi_xs.max())
    ax2.set_xlabel('photon energy (eV)')
    ax2.yaxis.get_major_formatter().set_powerlimits((0, 1))
    cbar2 = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin2, vmax=vmax2), cmap='inferno'), ax=ax2)
    cbar2.set_label('$\\frac{d^2 U}{d\\Omega d\\epsilon}$')
    cbar2.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    fig.savefig('strong_undulator.png', dpi=300)
    plt.close(fig)

def validate_weak_linear_plasma():

    plasma_density = 1.01e26

    K = 0.01
    gamma = 10000 / electron_rest_energy_mev
    lambda_u = np.sqrt(2 * np.pi * gamma / (classical_electron_radius * plasma_density))
    N_u = 10
    steps = 1000

    k_u = 2 * np.pi / lambda_u
    time_step = N_u * lambda_u / (steps * c_light)

    beta = np.sqrt(1 - gamma ** -2)
    a = K / (beta * gamma * k_u)
    trajectory = synchrad.track_particle_ion_collapse(a, 0, 0, 0, gamma, 1, plasma_density, 0, 0, 1e9, time_step, steps)

    energies, energies_midpoint, energies_step = synchrad.linspace_midpoint(1e5, 20e5, 200)
    phi_xs, phi_xs_midpoint, phi_xs_step = synchrad.linspace_midpoint(0, 5e-5, 200)
    result_numerical = synchrad.compute_radiation(trajectory, energies, phi_xs, np.array([0.0]), time_step)
    dd_numerical = np.sum(result_numerical ** 2, axis=3)
    dd_analytic = synchrad.undulator.weak_undulator_double_differential(energies, phi_xs, np.array([0.0]), K, k_u, N_u, gamma)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(plt.rcParams['figure.figsize'][0] * 1.5, plt.rcParams['figure.figsize'][1]))

    vmin1, vmax1 = dd_analytic.min(), dd_analytic.max()
    ax1.set_title('analytic')
    ax1.pcolormesh(energies_midpoint / 1e3, phi_xs_midpoint, dd_analytic[:, :, 0].T, vmin=vmin1, vmax=vmax1, cmap='inferno')
    ax1.set_xlim(energies.min() / 1e3, energies.max() / 1e3)
    ax1.set_ylim(phi_xs.min(), phi_xs.max())
    ax1.set_xlabel('photon energy (keV)')
    ax1.set_ylabel('$\\theta$ (rad)')
    ax1.yaxis.get_major_formatter().set_powerlimits((0, 1))
    cbar1 = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin1, vmax=vmax1), cmap='inferno'), ax=ax1)
    cbar1.set_label('$\\frac{d^2 U}{d\\Omega d\\epsilon}$')
    cbar1.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    vmin2, vmax2 = dd_numerical.min(), dd_numerical.max()
    ax2.set_title('numerical')
    ax2.pcolormesh(energies_midpoint / 1e3, phi_xs_midpoint, dd_numerical[:, :, 0].T, vmin=vmin2, vmax=vmax2, cmap='inferno')
    ax2.set_xlim(energies.min() / 1e3, energies.max() / 1e3)
    ax2.set_ylim(phi_xs.min(), phi_xs.max())
    ax2.set_xlabel('photon energy (keV)')
    ax2.yaxis.get_major_formatter().set_powerlimits((0, 1))
    cbar2 = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin2, vmax=vmax2), cmap='inferno'), ax=ax2)
    cbar2.set_label('$\\frac{d^2 U}{d\\Omega d\\epsilon}$')
    cbar2.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    fig.savefig('weak_linear_plasma.png', dpi=300)
    plt.close(fig)

def validate_strong_linear_plasma():

    plasma_density = 1.01e26

    K = 10
    gamma = 10000 / electron_rest_energy_mev
    lambda_u = np.sqrt(2 * np.pi * gamma / (classical_electron_radius * plasma_density))
    N_u = 10
    steps = 50000
    NW = 10

    k_u = 2 * np.pi / lambda_u
    time_step = N_u * lambda_u / (steps * c_light)

    beta = np.sqrt(1 - gamma ** -2)
    a = K / (beta * gamma * k_u)
    print(1e9 * a, 'nm')
    trajectory = synchrad.track_particle_ion_collapse(a, 0, 0, 0, gamma, 1, plasma_density, 0, 0, 1e9, time_step, steps)

    energies, energies_midpoint, energies_step = synchrad.linspace_midpoint(1, 5e7, 200)
    phi_xs, phi_xs_midpoint, phi_xs_step = synchrad.linspace_midpoint(0, 7e-4, 100)
    print('(this computation takes a while)')
    dd1 = synchrad.undulator.strong_undulator_double_differential(energies, phi_xs, np.array([0.0]), K, k_u, N_u, gamma, 1000, 1000)
    print('(ok we\'re done)')
    result_2 = synchrad.compute_radiation(trajectory, energies, phi_xs, np.array([0.0]), time_step)
    dd2 = np.sum(result_2 ** 2, axis=3)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(plt.rcParams['figure.figsize'][0] * 1.5, plt.rcParams['figure.figsize'][1]))

    vmin1, vmax1 = dd1.min(), dd1.max()
    ax1.set_title('analytic')
    ax1.pcolormesh(energies_midpoint, phi_xs_midpoint, dd1[:, :, 0].T, vmin=vmin1, vmax=vmax1, cmap='inferno')
    ax1.set_xlim(energies.min(), energies.max())
    ax1.set_ylim(phi_xs.min(), phi_xs.max())
    ax1.set_xlabel('photon energy (eV)')
    ax1.set_ylabel('$\\theta$ (rad)')
    ax1.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax1.yaxis.get_major_formatter().set_powerlimits((0, 1))
    cbar1 = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin1, vmax=vmax1), cmap='inferno'), ax=ax1)
    cbar1.set_label('$\\frac{d^2 U}{d\\Omega d\\epsilon}$')
    cbar1.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    vmin2, vmax2 = dd2.min(), dd2.max()
    ax2.set_title('numerical')
    ax2.pcolormesh(energies_midpoint, phi_xs_midpoint, dd2[:, :, 0].T, vmin=vmin2, vmax=vmax2, cmap='inferno')
    ax2.set_xlim(energies.min(), energies.max())
    ax2.set_ylim(phi_xs.min(), phi_xs.max())
    ax2.set_xlabel('photon energy (eV)')
    ax2.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax2.yaxis.get_major_formatter().set_powerlimits((0, 1))
    cbar2 = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin2, vmax=vmax2), cmap='inferno'), ax=ax2)
    cbar2.set_label('$\\frac{d^2 U}{d\\Omega d\\epsilon}$')
    cbar2.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    fig.savefig('strong_linear_plasma.png', dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    validate_beam_tracking()
    validate_weak_undulator()
    validate_strong_undulator()
    validate_weak_linear_plasma()
    validate_strong_linear_plasma()
