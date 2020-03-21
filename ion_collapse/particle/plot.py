from compute import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

vmin_distribution = None
vmax_distribution = None
double_differential_vmin = None
double_differential_vmax = None

spectra = []
total_energies = []

for j in range(len(x0_values)):
    print(int(sys.argv[1]), j)
    double_differential = np.fromfile(f'data/double_differential_{j}').reshape((len(energies), len(phi_xs), len(phi_ys)))
    distribution = np.sum(double_differential[:-1, :, :] * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
    if int(sys.argv[1]) == 0:
        spectra.append(np.sum(double_differential, axis=(1, 2)) * phi_xs_step * phi_ys_step)
    vmin_distribution = min(vmin_distribution, distribution.min()) if vmin_distribution is not None else distribution.min()
    double_differential_vmin = min(double_differential_vmin, double_differential.min()) if double_differential_vmin is not None else double_differential.min()
    vmax_distribution = max(vmax_distribution, distribution.max()) if vmax_distribution is not None else distribution.max()
    double_differential_vmax = max(double_differential_vmax, double_differential.max()) if double_differential_vmax is not None else double_differential.max()

double_differential = np.fromfile(f'data/double_differential_{i}').reshape((len(energies), len(phi_xs), len(phi_ys)))
distribution = np.sum(double_differential[:-1, :, :] * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)

fig, ax = plt.subplots()
ax.set_title('$r_0$ = {:.2f} nm'.format(x0 * 1e9))
ax.pcolormesh(phi_xs_midpoint, phi_ys_midpoint, distribution.T, cmap='inferno', vmin=vmin_distribution, vmax=vmax_distribution)
ax.set_xlim(phi_xs.min(), phi_xs.max())
ax.set_ylim(phi_ys.min(), phi_ys.max())
ax.set_xlabel('$\\phi_x$ (rad)')
ax.set_ylabel('$\\phi_y$ (rad)')
ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin_distribution, vmax=vmax_distribution), cmap='inferno'), ax=ax)
cbar.set_label('$\\frac{dU}{d\\Omega}$ (eV)')
cbar.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
fig.savefig(f'frames/distribution_{i}.png', dpi=300)
plt.close(fig)

horizontal_slice = double_differential[:, :, zero_index]
vertical_slice = double_differential[:, zero_index, :]

fig, ax = plt.subplots()
ax.set_title('$r_0$ = {:.2f} nm'.format(x0 * 1e9))
ax.pcolormesh(energies_midpoint, phi_xs_midpoint, horizontal_slice.T, vmin=double_differential_vmin, vmax=double_differential_vmax, cmap='inferno')
ax.set_xlim(energies.min(), energies.max())
ax.set_ylim(phi_xs_midpoint.min(), phi_xs_midpoint.max())
ax.set_xscale('log')
ax.set_xlabel('photon energy (eV)')
ax.set_ylabel('$\\phi_x$ (rad)')
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=double_differential_vmin, vmax=double_differential_vmax), cmap='inferno'), ax=ax)
cbar.set_label('$\\frac{d^2 U}{d\\Omega d\\epsilon}$')
cbar.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
fig.savefig(f'frames/double_differential_slice_x_{i}.png', dpi=300)
plt.close(fig)

fig, ax = plt.subplots()
ax.set_title('$r_0$ = {:.2f} nm'.format(x0 * 1e9))
ax.pcolormesh(energies_midpoint, phi_ys_midpoint, vertical_slice.T, vmin=double_differential_vmin, vmax=double_differential_vmax, cmap='inferno')
ax.set_xlim(energies.min(), energies.max())
ax.set_ylim(phi_ys_midpoint.min(), phi_ys_midpoint.max())
ax.set_xscale('log')
ax.set_xlabel('photon energy (eV)')
ax.set_ylabel('$\\phi_y$ (rad)')
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=double_differential_vmin, vmax=double_differential_vmax), cmap='inferno'), ax=ax)
cbar.set_label('$\\frac{d^2 U}{d\\Omega d\\epsilon}$')
cbar.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
fig.savefig(f'frames/double_differential_slice_y_{i}.png', dpi=300)
plt.close(fig)

if i == 0:

    spectrum = np.array(spectra)

    fig, ax = plt.subplots()
    for j in [np.abs(x0_values - target * 1e-9).argmin() for target in (20, 40, 60, 80, 100, 120, 140, 160, 180, 200)]:
        ax.plot(energies, spectrum[j], label='$x_0$ = {:.2f} nm'.format(x0_values[j] * 1e9))
    ax.set_xlabel('energy (eV)')
    ax.set_ylabel('$\\frac{dU}{d\\epsilon}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig('results/spectrum.png', dpi=300)
    plt.close(fig)

    total_energy = np.sum(spectrum[:, :-1] * (energies[1:] - energies[:-1])[np.newaxis, :], axis=1)
    fig, ax = plt.subplots()
    ax.plot(x0_values * 1e9, total_energy)
    ax.set_ylabel('energy radiated (eV)')
    ax.set_xlabel('$x_0$ (nm)')
    fig.savefig('results/total_energy.png', dpi=300)
    plt.close(fig)
