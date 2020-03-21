import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np

c_light = 299792458

def trajectory(t, trajectories, filename):
    assert trajectories.shape[0] == 1
    x = trajectories[0, :, 0]
    y = trajectories[0, :, 1]
    bx = trajectories[0, :, 3]
    by = trajectories[0, :, 4]
    r = np.sqrt(x ** 2 + y ** 2)
    br = (x * bx + y * by) / r
    z_cm = 100 * (trajectories[0, :, 2] + c_light * t)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax1.plot(z_cm, x, label='x (m)')
    ax1.plot(z_cm, y, label='y (m)')
    ax1.plot(z_cm, r, label='r (m)')
    ax1.get_yaxis().get_major_formatter().set_powerlimits((0, 1))
    ax1.legend()
    ax2.plot(z_cm, bx, label='$\\beta_x$')
    ax2.plot(z_cm, by, label='$\\beta_y$')
    ax2.plot(z_cm, br, label='$\\beta_r$')
    ax2.get_yaxis().get_major_formatter().set_powerlimits((0, 1))
    ax2.legend()
    ax2.set_xlabel('z (cm)')
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def phase_space(trajectories, filename):
    assert trajectories.shape[0] == 1
    x = trajectories[0, :, 0]
    y = trajectories[0, :, 1]
    bx = trajectories[0, :, 3]
    by = trajectories[0, :, 4]
    g = trajectories[0, :, 5]
    r = np.sqrt(x ** 2 + y ** 2)
    th = np.arctan2(y, x)
    br = (x * bx + y * by) / r
    bth = (x * by - y * bx) / (x ** 2 + y ** 2)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(plt.rcParams['figure.figsize'][0] * 2, plt.rcParams['figure.figsize'][1] * 1))
    ax1.plot(x, g * bx)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('$\\gamma \\beta_x$')
    ax1.get_xaxis().get_major_formatter().set_powerlimits((0, 2))
    ax1.get_yaxis().get_major_formatter().set_powerlimits((0, 2))
    ax2.plot(y, g * by)
    ax2.set_xlabel('y (m)')
    ax2.set_ylabel('$\\gamma \\beta_y$')
    ax2.get_xaxis().get_major_formatter().set_powerlimits((0, 2))
    ax2.get_yaxis().get_major_formatter().set_powerlimits((0, 2))
    ax3.plot(r, g * br)
    ax3.set_xlabel('r (m)')
    ax3.set_ylabel('$\\gamma \\beta_r$')
    ax3.get_xaxis().get_major_formatter().set_powerlimits((0, 2))
    ax3.get_yaxis().get_major_formatter().set_powerlimits((0, 2))
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def transverse(trajectories, filename):
    assert trajectories.shape[0] == 1
    fig, ax = plt.subplots()
    limit = 1.05 * np.sqrt(trajectories[0, :, 0] ** 2 + trajectories[0, :, 1] ** 2).max()
    ax.plot(trajectories[0, :, 0], trajectories[0, :, 1])
    ax.plot(trajectories[0, 0, 0], trajectories[0, 0, 1], 'ro')
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.get_xaxis().get_major_formatter().set_powerlimits((0, 1))
    ax.get_yaxis().get_major_formatter().set_powerlimits((0, 1))
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def distribution(double_differential, energies, phi_xs, phi_xs_midpoint, phi_ys, phi_ys_midpoint, filename, title=None):
    distribution = np.sum((double_differential[:-1, :, :] * (energies[1:, np.newaxis, np.newaxis] - energies[:-1, np.newaxis, np.newaxis])), axis=0)
    vmin, vmax = distribution.min(), distribution.max()
    fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    ax.pcolormesh(phi_xs_midpoint, phi_ys_midpoint, distribution.T, cmap='inferno', vmin=vmin, vmax=vmax)
    ax.set_xlim(phi_xs.min(), phi_xs.max())
    ax.set_ylim(phi_ys.min(), phi_ys.max())
    ax.set_xlabel('$\\phi_x$ (rad)')
    ax.set_ylabel('$\\phi_y$ (rad)')
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap='inferno'), ax=ax)
    cbar.set_label('$\\frac{dU}{d\\Omega}$ (eV)')
    cbar.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def spectrum(double_differential, energies, phi_xs_step, phi_ys_step, filename):
    assert double_differential.ndim == 3
    fig, ax = plt.subplots()
    ax.plot(energies, np.sum(double_differential, axis=(1, 2)) * phi_xs_step * phi_ys_step)
    ax.set_xlabel('energy (eV)')
    ax.set_ylabel('$\\frac{dU}{d\\epsilon}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(filename, dpi=300)
    plt.close(fig)
