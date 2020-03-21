from compute import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import scipy.integrate

electron_rest_energy_mev = 0.510998
c_light = 299792458
classical_electron_radius = 2.8179403227e-15

t = time_step * np.arange(steps + 1)

trajectories = np.empty((128 * particles, steps + 1, 9), dtype=np.float64)
trajectories = trajectories[:, ::100, :]
trajectories = np.copy(trajectories)

for j in range(128):
    mm = np.memmap(f'data/trajectory_{j}', dtype=np.float64, shape=(particles, steps + 1, 9))
    trajectories[j*particles:(j+1)*particles, :, :] = mm[:, ::100, :]

gamma_final = gamma_initial + (-plasma_length * accelerating_field / (electron_rest_energy_mev * 1e6))
bennett_radius_final = bennett_radius_initial * (gamma_final / gamma_initial) ** -0.25
sigma_r_initial = 0.5 * bennett_radius_initial * np.sqrt(rho_ion / plasma_density)
sigma_r_dot_initial = bennett_radius_initial * c_light * np.sqrt(np.pi * classical_electron_radius * ion_atomic_number * rho_ion / (2 * gamma_initial))
sigma_r_dot_final = bennett_radius_final * c_light * np.sqrt(np.pi * classical_electron_radius * ion_atomic_number * rho_ion / (2 * gamma_final))

n_bins = 50

r_max = 5 * bennett_radius_initial
v_max = 5 * sigma_r_dot_initial
r_ticklabels = [r'$0$', r'$5a_{\mathrm{initial}}$']
v_ticklabels = [r'$-5\sigma_{\dot{r}, \mathrm{initial}}$', r'$0$', r'5$\sigma_{\dot{r}, \mathrm{initial}}$']

r_linspace = np.linspace(0, r_max, 1000)
v_linspace = np.linspace(-v_max, v_max, 1000)

def get_r_pdf(sigma_r, bennett_radius):
    r_pdf_unnormalized = lambda r: r * np.exp(-r ** 2 / (2 * sigma_r ** 2)) / ((1 + ((r / bennett_radius) ** 2)) ** 2)
    normalization_factor = scipy.integrate.quad(r_pdf_unnormalized, 0, r_max)[0]
    return lambda r: r_pdf_unnormalized(r) / normalization_factor

r_height = 1.5 * get_r_pdf(sigma_r_initial, bennett_radius_initial)(r_linspace).max()
th_height = 1.5 / (2 * np.pi)
v_height = 1.5 / (np.sqrt(2 * np.pi) * sigma_r_dot_initial)

frame_indices = np.arange(steps + 1)[::100]

for i in np.arange(len(frame_indices))[int(sys.argv[1])::128]:

    w = frame_indices[i]

    print(i, w)

    z = (w / steps) * plasma_length
    gamma_current = gamma_initial + (-z * accelerating_field / (electron_rest_energy_mev * 1e6))
    bennett_radius_current = bennett_radius_initial * (gamma_current / gamma_initial) ** -0.25
    sigma_r_current = 0.5 * bennett_radius_current * np.sqrt(rho_ion / plasma_density)
    sigma_r_dot_current = bennett_radius_current * c_light * np.sqrt(np.pi * classical_electron_radius * ion_atomic_number * rho_ion / (2 * gamma_current))

    x = trajectories[:, i, 0]
    y = trajectories[:, i, 1]
    vx = trajectories[:, i, 3] * c_light
    vy = trajectories[:, i, 4] * c_light
    r = np.sqrt(x ** 2 + y ** 2)
    th = np.arctan2(y, x)
    vr = (x * vx + y * vy) / r
    rvth = (x * vy - y * vx) / r

    r = r[r < r_max]
    vr = vr[np.logical_and(-v_max <= vr, vr <= v_max)]
    rvth = rvth[np.logical_and(-v_max <= rvth, rvth <= v_max)]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(6.4, 4.8))

    fig.suptitle('Beam Distribution, z = {:.2f} cm'.format(z * 100))

    ax1.hist(r, density=True, bins=n_bins, range=(0, r_max))
    ax1.plot(r_linspace, get_r_pdf(sigma_r_current, bennett_radius_current)(r_linspace))
    ax1.set_xlabel(r'$r$')
    ax1.set_xticks([0, r_max])
    ax1.set_xticklabels(r_ticklabels)
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    ax1.set_ylim(0, r_height)

    ax2.hist(th, density=True, bins=n_bins, range=(-np.pi, np.pi))
    ax2.plot([-np.pi, np.pi], [1 / (2 * np.pi), 1 / (2 * np.pi)])
    ax2.set_xlabel(r'$\theta$')
    ax2.set_xticks([-np.pi, 0, np.pi])
    ax2.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_ylim(0, th_height)

    v_analytic = np.exp(-v_linspace ** 2 / (2 * sigma_r_dot_current ** 2)) / (np.sqrt(2 * np.pi) * sigma_r_dot_current)

    ax3.hist(vr, density=True, bins=n_bins, range=(-v_max, v_max))
    ax3.plot(v_linspace, v_analytic)
    ax3.set_xlabel(r'$\dot{r}$')
    ax3.set_xticks([-v_max, 0, v_max])
    ax3.set_xticklabels(v_ticklabels)
    ax3.set_yticks([])
    ax3.set_yticklabels([])
    ax3.set_ylim(0, v_height)

    ax4.hist(rvth, density=True, bins=n_bins, range=(-v_max, v_max))
    ax4.plot(v_linspace, v_analytic)
    ax4.set_xlabel(r'$r \dot{\theta}$')
    ax4.set_xticks([-v_max, 0, v_max])
    ax4.set_xticklabels(v_ticklabels)
    ax4.set_yticks([])
    ax4.set_yticklabels([])
    ax4.set_ylim(0, v_height)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'frames/histogram_{i}.png', dpi=300)
    plt.close(fig)

print(sys.argv[1], 'done!')
