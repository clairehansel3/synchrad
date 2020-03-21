from compute import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

result = np.zeros((len(energies), len(thetas), 1, 6))

for i in range(128):
    result += np.fromfile(f'data/radiation_{i}').reshape(result.shape)

double_differential = np.sum(result ** 2, axis=3)

fig, ax = plt.subplots()
vmin, vmax = double_differential.min(), double_differential.max()
ax.pcolormesh(energies_midpoint, thetas_midpoint * 1e3, double_differential[:, :, 0].T / (128 * particles), vmin=vmin / (128 * particles), vmax=vmax / (128 * particles), cmap='inferno')
ax.set_xlim(energies.min(), energies.max())
ax.set_ylim(thetas.min() * 1e3, thetas.max() * 1e3)
ax.set_xscale('log')
ax.set_xlabel('photon energy (eV)')
ax.set_ylabel(r'$\theta$ (mrad)')
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap='inferno'), ax=ax)
cbar.set_label(r'$\frac{d^2 U}{d\Omega d\epsilon}$ per particle')
cbar.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
fig.savefig('double_differential.png', dpi=300)
plt.close(fig)

spectrum = 2 * np.pi * np.sum(double_differential[:, :, 0] * thetas[np.newaxis, :], axis=1) * thetas_step

fig, ax = plt.subplots()
ax.plot(energies, spectrum / (128 * particles))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('photon energy (eV)')
ax.set_ylabel(r'$\frac{dU}{d\epsilon}$ per particle')
fig.savefig('spectrum.png', dpi=300)
plt.close(fig)

distribution = np.sum(double_differential[:-1, :, 0] * (energies[1:, np.newaxis] - energies[:-1, np.newaxis]), axis=0)

fig, ax = plt.subplots()
ax.plot(thetas * 1e3, distribution / (128 * particles))
ax.set_yscale('log')
ax.set_xlabel(r'$\theta$ (mrad)')
ax.set_ylabel(r'$\frac{dU}{d\Omega}$ per particle (eV)')
fig.savefig('distribution.png', dpi=300)
plt.close(fig)

total_energy = 2 * np.pi * np.sum(distribution * thetas) * thetas_step
with open('result.txt', 'w+') as f:
    f.write('total energy = {:.5e} eV'.format(total_energy))
