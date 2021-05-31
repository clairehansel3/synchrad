#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>

struct Particle {
  double x, y, zeta, bx, by, g, bxd, byd, gd;
};

struct EnergyPhixPhiy {
  double energy, phi_x, phi_y;
};

extern "C"
{
void track_particle_bennett(double* result, double x0, double y0, double vx0, double vy0,
  double gamma_initial, double ion_atomic_number, double plasma_density,
  double rho_ion, double accelerating_field, double bennett_radius_initial,
  double time_step, std::size_t steps);

void track_particle_linear(double* result, double x0, double y0, double vx0, double vy0,
  double gamma_initial, double ion_atomic_number, double plasma_density,
  double accelerating_field, double time_step, std::size_t steps);

void compute_radiation_grid(double* __restrict__ result, Particle* __restrict__ data,
  double* __restrict__ energies, double* __restrict__ phi_xs,
  double* __restrict__ phi_ys, std::size_t n_particles, std::size_t n_steps,
  std::size_t n_energies, std::size_t n_phi_xs, std::size_t n_phi_ys,
  double time_step);

void compute_radiation_list(double* __restrict__ result,
  Particle* __restrict__ data, EnergyPhixPhiy* __restrict__ inputs,
  std::size_t n_particles, std::size_t n_steps, std::size_t n_inputs,
  double time_step);

void compute_radiation_grid_nan(double* __restrict__ result, Particle* __restrict__ data,
  double* __restrict__ energies, double* __restrict__ phi_xs,
  double* __restrict__ phi_ys, std::size_t n_particles, std::size_t n_steps,
  std::size_t n_energies, std::size_t n_phi_xs, std::size_t n_phi_ys,
  double time_step);

void compute_radiation_list_nan(double* __restrict__ result,
  Particle* __restrict__ data, EnergyPhixPhiy* __restrict__ inputs,
  std::size_t n_particles, std::size_t n_steps, std::size_t n_inputs,
  double time_step);
}

[[noreturn]] static void new_signalhandler(int signum)
{
  std::cerr << "interrupt recieved (" << signum << "), exiting\n";
  std::exit(signum);
}

class SignalHandlerHelper {
public:
  SignalHandlerHelper();
  ~SignalHandlerHelper();
private:
  void (*m_previous_signalhandler)(int);
};

SignalHandlerHelper::SignalHandlerHelper()
{
  m_previous_signalhandler = std::signal(SIGINT, new_signalhandler);
}

SignalHandlerHelper::~SignalHandlerHelper()
{
  std::signal(SIGINT, m_previous_signalhandler);
}

static constexpr double elementary_charge = 1.60217662e-19;
static constexpr double vacuum_permittivity = 8.8541878128e-12;
static constexpr double electron_mass = 9.1093837015e-31;
static constexpr double c_light = 299792458.0;
static constexpr double hbar_ev = 6.582119569e-16; // eV * s
static constexpr double constant = 0.013595738304; // sqrt(e^2 / (16 pi^3 epsilon_0 hbar c))

template <unsigned long n>
static std::array<double, n> arr_add(std::array<double, n> const& a, std::array<double, n> const& b)
{
  std::array<double, n> c;
  for (std::size_t i = 0; i != n; ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

template <unsigned long n>
static std::array<double, n> arr_mul(double a, std::array<double, n> const& b)
{
  std::array<double, n> c;
  for (std::size_t i = 0; i != n; ++i) {
    c[i] = a * b[i];
  }
  return c;
}

template <unsigned long n>
static void rk4(std::array<double, n> y0, std::function<std::array<double, n>(double, std::array<double, n>)> f, std::function<void(double, std::array<double, n>)> write, std::size_t steps, double time_step)
{
  std::array<double, n> y = y0;
  write(0.0, y);
  for (std::size_t step = 0; step != steps; ++step) {
    double t = step * time_step;

    #ifdef EULER

    // euler method
    y = arr_add(y, arr_mul(time_step, f(t, y)));

    #else

    // 4th order runge-kutta
    std::array<double, n> k1 = arr_mul(time_step, f(t, y));
    std::array<double, n> k2 = arr_mul(time_step, f(t + 0.5 * time_step, arr_add(y, arr_mul(0.5, k1))));
    std::array<double, n> k3 = arr_mul(time_step, f(t + 0.5 * time_step, arr_add(y, arr_mul(0.5, k2))));
    std::array<double, n> k4 = arr_mul(time_step, f(t + time_step, arr_add(y, k3)));
    y = arr_add(y, arr_mul(1/6.0, arr_add(arr_add(k1, k4), arr_mul(2.0, arr_add(k2, k3)))));

    #endif

    write(t + time_step, y);
  }
}

void track_particle_bennett(double* result, double x0, double y0, double vx0, double vy0,
  double gamma_initial, double ion_atomic_number, double plasma_density,
  double rho_ion, double accelerating_field, double bennett_radius_initial,
  double time_step, std::size_t steps)
{
  SignalHandlerHelper signalhandlerhelper{};

  std::array<double, 5> initial_coordinates;
  initial_coordinates[0] = x0;
  initial_coordinates[1] = y0;
  initial_coordinates[2] = 0;
  initial_coordinates[3] = gamma_initial * vx0 / c_light;
  initial_coordinates[4] = gamma_initial * vy0 / c_light;

  double a = (1 + std::pow(initial_coordinates[3], 2) + std::pow(initial_coordinates[4], 2)) * std::pow(gamma_initial, -2);

  double constant_1 = ion_atomic_number * std::pow(elementary_charge, 2) * plasma_density / (2 * vacuum_permittivity * electron_mass * c_light);
  double constant_2 = rho_ion / plasma_density;
  double constant_3 = std::pow(bennett_radius_initial, -2) * std::pow(gamma_initial, -0.5);
  double constant_4 = std::sqrt(std::pow(gamma_initial, 2) - std::pow(initial_coordinates[3], 2) - std::pow(initial_coordinates[4], 2) - 1);
  double constant_5 = -elementary_charge * accelerating_field / (electron_mass * c_light);
  double constant_6 = gamma_initial * (-0.5 * a - 0.125 * a * a - 0.0625 * a * a * a * a);

  std::function<std::array<double, 5>(double, std::array<double, 5>)> f = [constant_1, constant_2, constant_3, constant_4, constant_5, constant_6, gamma_initial](double t, std::array<double, 5> coordinates){
   double gamma = std::sqrt(1 + std::pow(coordinates[3], 2) + std::pow(coordinates[4], 2) + std::pow(constant_4 + constant_5 * t, 2));
   double value = -constant_1 * (1 + constant_2  / (1 + constant_3 * std::sqrt(gamma) * (std::pow(coordinates[0], 2) + std::pow(coordinates[1], 2))));
   std::array<double, 5> rhs;
   rhs[0] = coordinates[3] * c_light / gamma;
   rhs[1] = coordinates[4] * c_light / gamma;
   rhs[2] = c_light * (constant_6 + constant_5 * t + (gamma_initial - gamma)) / gamma;
   rhs[3] = value * coordinates[0];
   rhs[4] = value * coordinates[1];
   return rhs;
  };

  std::function<void(double, std::array<double, 5>)> write = [&result, constant_1, constant_2, constant_3, constant_4, constant_5](double t, std::array<double, 5> coordinates){
    double gamma = std::sqrt(1 + std::pow(coordinates[3], 2) + std::pow(coordinates[4], 2) + std::pow(constant_4 + constant_5 * t, 2));
    double bx = coordinates[3] / gamma;
    double by = coordinates[4] / gamma;
    double value = -constant_1 * (1 + constant_2  / (1 + constant_3 * std::sqrt(gamma) * (std::pow(coordinates[0], 2) + std::pow(coordinates[1], 2))));
    double gamma_dot = (constant_5 * (constant_4 + constant_5 * t) + value * (coordinates[0] * coordinates[3] + coordinates[1] * coordinates[4])) / gamma;
    double bxd = (value * coordinates[0] / gamma) - (coordinates[3] * gamma_dot * std::pow(gamma, -2));
    double byd = (value * coordinates[1] / gamma) - (coordinates[4] * gamma_dot * std::pow(gamma, -2));
    std::array<double, 9> new_coordinates{coordinates[0], coordinates[1], coordinates[2], bx, by, gamma, bxd, byd, gamma_dot};
    std::copy_n(new_coordinates.data(), 9, result);
    result += 9;
  };

  rk4(initial_coordinates, f, write, steps, time_step);
}

void track_particle_linear(double* result, double x0, double y0, double vx0, double vy0,
  double gamma_initial, double ion_atomic_number, double plasma_density,
  double accelerating_field, double time_step, std::size_t steps)
{
  SignalHandlerHelper signalhandlerhelper{};

  std::array<double, 5> initial_coordinates;
  initial_coordinates[0] = x0;
  initial_coordinates[1] = y0;
  initial_coordinates[2] = 0;
  initial_coordinates[3] = gamma_initial * vx0 / c_light;
  initial_coordinates[4] = gamma_initial * vy0 / c_light;

  double a = (1 + std::pow(initial_coordinates[3], 2) + std::pow(initial_coordinates[4], 2)) * std::pow(gamma_initial, -2);
  double constant_1 = ion_atomic_number * std::pow(elementary_charge, 2) * plasma_density / (2 * vacuum_permittivity * electron_mass * c_light);
  double constant_4 = std::sqrt(std::pow(gamma_initial, 2) - std::pow(initial_coordinates[3], 2) - std::pow(initial_coordinates[4], 2) - 1);
  double constant_5 = -elementary_charge * accelerating_field / (electron_mass * c_light);
  double constant_6 = gamma_initial * (-0.5 * a - 0.125 * a * a - 0.0625 * a * a * a * a);

  std::function<std::array<double, 5>(double, std::array<double, 5>)> f = [constant_1, constant_4, constant_5, constant_6, gamma_initial](double t, std::array<double, 5> coordinates){
   double gamma = std::sqrt(1 + std::pow(coordinates[3], 2) + std::pow(coordinates[4], 2) + std::pow(constant_4 + constant_5 * t, 2));
   std::array<double, 5> rhs;
   rhs[0] = coordinates[3] * c_light / gamma;
   rhs[1] = coordinates[4] * c_light / gamma;
   rhs[2] = c_light * (constant_6 + constant_5 * t + (gamma_initial - gamma)) / gamma;
   rhs[3] = -constant_1 * coordinates[0];
   rhs[4] = -constant_1 * coordinates[1];
   return rhs;
  };

  std::function<void(double, std::array<double, 5>)> write = [&result, constant_1, constant_4, constant_5](double t, std::array<double, 5> coordinates){
    double gamma = std::sqrt(1 + std::pow(coordinates[3], 2) + std::pow(coordinates[4], 2) + std::pow(constant_4 + constant_5 * t, 2));
    double bx = coordinates[3] / gamma;
    double by = coordinates[4] / gamma;
    double gamma_dot = (constant_5 * (constant_4 + constant_5 * t) -constant_1 * (coordinates[0] * coordinates[3] + coordinates[1] * coordinates[4])) / gamma;
    double bxd = (-constant_1 * coordinates[0] / gamma) - (coordinates[3] * gamma_dot * std::pow(gamma, -2));
    double byd = (-constant_1 * coordinates[1] / gamma) - (coordinates[4] * gamma_dot * std::pow(gamma, -2));
    std::array<double, 9> new_coordinates{coordinates[0], coordinates[1], coordinates[2], bx, by, gamma, bxd, byd, gamma_dot};
    std::copy_n(new_coordinates.data(), 9, result);
    result += 9;
  };

  rk4(initial_coordinates, f, write, steps, time_step);
}

static std::array<double, 3> cross(std::array<double, 3> a, std::array<double, 3> b)
{
  return {
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  };
}

static std::array<double, 3> subtract(std::array<double, 3> a, std::array<double, 3> b)
{
  return {
    a[0] - b[0],
    a[1] - b[1],
    a[2] - b[2]
  };
}

static double dot(std::array<double, 3> a, std::array<double, 3> b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void compute_radiation_grid(double* __restrict__ result, Particle* __restrict__ data,
  double* __restrict__ energies, double* __restrict__ phi_xs,
  double* __restrict__ phi_ys, std::size_t n_particles, std::size_t n_steps,
  std::size_t n_energies, std::size_t n_phi_xs, std::size_t n_phi_ys,
  double time_step)
{
  SignalHandlerHelper signalhandlerhelper{};

  //std::size_t i = 0;
  //int percent = -1;

  for (std::size_t j = 0; j != n_energies; ++j) {
    double frequency = energies[j] / hbar_ev;
    for (std::size_t k = 0; k != n_phi_xs; ++k) {
      double phi_x = phi_xs[k];
      for (std::size_t l = 0; l != n_phi_ys; ++l) {
        double phi_y = phi_ys[l];

        std::size_t result_index = 6 * (l + n_phi_ys * (k + n_phi_xs * j));
        for (std::size_t a = 0; a != 6; ++a)
          result[result_index + a] = 0.0;

        for (std::size_t m = 0; m != n_particles; ++m) {
          for (std::size_t o = 0; o != n_steps + 1; ++o) {

            /*
            int new_percent = static_cast<int>((100 * i) / (n_energies * n_phi_xs * n_phi_ys * n_particles * (n_steps + 1)));
            if (percent != new_percent) {
              percent = new_percent;
              std::cout << percent << '%' << std::endl;
            }
            ++i;
            */

            double t = time_step * o;

            Particle particle = data[o + m * (n_steps + 1)];

            std::array<double, 3> n;
            std::array<double, 3> b;
            std::array<double, 3> bd;
            n[0] = std::sin(phi_x);
            n[1] = std::sin(phi_y);
            n[2] = std::sqrt(1 - std::pow(n[0], 2) - std::pow(n[1], 2));
            b[0] = particle.bx;
            b[1] = particle.by;
            b[2] = std::sqrt(1 - std::pow(particle.g, -2) - std::pow(particle.bx, 2) - std::pow(particle.by, 2));
            bd[0] = particle.bxd;
            bd[1] = particle.byd;
            bd[2] = -(particle.gd * std::pow(particle.g, -3) + particle.bx * particle.bxd + particle.by * particle.byd) / b[2];
            auto vector = cross(n, cross(subtract(n, b), bd));
            auto denom = std::pow(1 - dot(b, n), -2);
            double n_transverse2 = std::pow(n[0], 2) + std::pow(n[1], 2);
            double value = 0.5 * n_transverse2 + 0.125 * std::pow(n_transverse2, 2) + 0.0625 * std::pow(n_transverse2, 3);
            double phase = frequency * ((t * value) - (n[0] * particle.x + n[1] * particle.y + n[2] * particle.zeta) / c_light);
            double exponential_real = std::cos(phase);
            double exponential_imag = std::sin(phase);
            double integration_multiplier = (o == 0 || o == n_steps) ? 0.5 : 1.0;

            result[result_index + 0] += vector[0] * exponential_real * denom * integration_multiplier * time_step * constant;
            result[result_index + 1] += vector[0] * exponential_imag * denom * integration_multiplier * time_step * constant;
            result[result_index + 2] += vector[1] * exponential_real * denom * integration_multiplier * time_step * constant;
            result[result_index + 3] += vector[1] * exponential_imag * denom * integration_multiplier * time_step * constant;
            result[result_index + 4] += vector[2] * exponential_real * denom * integration_multiplier * time_step * constant;
            result[result_index + 5] += vector[2] * exponential_imag * denom * integration_multiplier * time_step * constant;
          }
        }
      }
    }
  }
}

void compute_radiation_list(double* __restrict__ result,
  Particle* __restrict__ data, EnergyPhixPhiy* __restrict__ inputs,
  std::size_t n_particles, std::size_t n_steps, std::size_t n_inputs,
  double time_step)
{
  SignalHandlerHelper signalhandlerhelper{};

  //std::size_t i = 0;
  //int percent = -1;

  for (std::size_t j = 0; j != n_inputs; ++j) {
    double frequency = inputs[j].energy / hbar_ev;
    double phi_x = inputs[j].phi_x;
    double phi_y = inputs[j].phi_y;

    std::size_t result_index = 6 * j;

    for (std::size_t a = 0; a != 6; ++a)
      result[result_index + a] = 0.0;

    for (std::size_t m = 0; m != n_particles; ++m) {
      for (std::size_t o = 0; o != n_steps + 1; ++o) {
        /*
        int new_percent = static_cast<int>((100 * i) / (n_inputs * n_particles * (n_steps + 1)));
        if (percent != new_percent) {
          percent = new_percent;
          std::cout << percent << '%' << std::endl;
        }
        ++i;
        */

        double t = time_step * o;

        Particle particle = data[o + m * (n_steps + 1)];

        std::array<double, 3> n;
        std::array<double, 3> b;
        std::array<double, 3> bd;
        n[0] = std::sin(phi_x);
        n[1] = std::sin(phi_y);
        n[2] = std::sqrt(1 - std::pow(n[0], 2) - std::pow(n[1], 2));
        b[0] = particle.bx;
        b[1] = particle.by;
        b[2] = std::sqrt(1 - std::pow(particle.g, -2) - std::pow(particle.bx, 2) - std::pow(particle.by, 2));
        bd[0] = particle.bxd;
        bd[1] = particle.byd;
        bd[2] = -(particle.gd * std::pow(particle.g, -3) + particle.bx * particle.bxd + particle.by * particle.byd) / b[2];
        auto vector = cross(n, cross(subtract(n, b), bd));
        auto denom = std::pow(1 - dot(b, n), -2);
        double n_transverse2 = std::pow(n[0], 2) + std::pow(n[1], 2);
        double value = 0.5 * n_transverse2 + 0.125 * std::pow(n_transverse2, 2) + 0.0625 * std::pow(n_transverse2, 3);
        double phase = frequency * ((t * value) - (n[0] * particle.x + n[1] * particle.y + n[2] * particle.zeta) / c_light);
        double exponential_real = std::cos(phase);
        double exponential_imag = std::sin(phase);
        double integration_multiplier = (o == 0 || o == n_steps) ? 0.5 : 1.0;

        result[result_index + 0] += vector[0] * exponential_real * denom * integration_multiplier * time_step * constant;
        result[result_index + 1] += vector[0] * exponential_imag * denom * integration_multiplier * time_step * constant;
        result[result_index + 2] += vector[1] * exponential_real * denom * integration_multiplier * time_step * constant;
        result[result_index + 3] += vector[1] * exponential_imag * denom * integration_multiplier * time_step * constant;
        result[result_index + 4] += vector[2] * exponential_real * denom * integration_multiplier * time_step * constant;
        result[result_index + 5] += vector[2] * exponential_imag * denom * integration_multiplier * time_step * constant;
      }
    }
  }
}

void compute_radiation_grid_nan(double* __restrict__ result, Particle* __restrict__ data,
  double* __restrict__ energies, double* __restrict__ phi_xs,
  double* __restrict__ phi_ys, std::size_t n_particles, std::size_t n_steps,
  std::size_t n_energies, std::size_t n_phi_xs, std::size_t n_phi_ys,
  double time_step)
{
  SignalHandlerHelper signalhandlerhelper{};

  //std::size_t i = 0;
  //int percent = -1;

  for (std::size_t j = 0; j != n_energies; ++j) {
    double frequency = energies[j] / hbar_ev;
    for (std::size_t k = 0; k != n_phi_xs; ++k) {
      double phi_x = phi_xs[k];
      for (std::size_t l = 0; l != n_phi_ys; ++l) {
        double phi_y = phi_ys[l];

        std::size_t result_index = 6 * (l + n_phi_ys * (k + n_phi_xs * j));
        for (std::size_t a = 0; a != 6; ++a)
          result[result_index + a] = 0.0;

        for (std::size_t m = 0; m != n_particles; ++m) {
          bool last_was_nan = true;
          for (std::size_t o = 0; o != n_steps + 1; ++o) {
            
            /*
            int new_percent = static_cast<int>((100 * i) / (n_energies * n_phi_xs * n_phi_ys * n_particles * (n_steps + 1)));
            if (percent != new_percent) {
              percent = new_percent;
              std::cout << percent << '%' << std::endl;
            }
            ++i;
            */

            double t = time_step * o;

            Particle particle = data[o + m * (n_steps + 1)];
            
            if (last_was_nan && std::isnan(particle.x))
              continue;

            std::array<double, 3> n;
            std::array<double, 3> b;
            std::array<double, 3> bd;
            n[0] = std::sin(phi_x);
            n[1] = std::sin(phi_y);
            n[2] = std::sqrt(1 - std::pow(n[0], 2) - std::pow(n[1], 2));
            b[0] = particle.bx;
            b[1] = particle.by;
            b[2] = std::sqrt(1 - std::pow(particle.g, -2) - std::pow(particle.bx, 2) - std::pow(particle.by, 2));
            bd[0] = particle.bxd;
            bd[1] = particle.byd;
            bd[2] = -(particle.gd * std::pow(particle.g, -3) + particle.bx * particle.bxd + particle.by * particle.byd) / b[2];
            auto vector = cross(n, cross(subtract(n, b), bd));
            auto denom = std::pow(1 - dot(b, n), -2);
            double n_transverse2 = std::pow(n[0], 2) + std::pow(n[1], 2);
            double value = 0.5 * n_transverse2 + 0.125 * std::pow(n_transverse2, 2) + 0.0625 * std::pow(n_transverse2, 3);
            double phase = frequency * ((t * value) - (n[0] * particle.x + n[1] * particle.y + n[2] * particle.zeta) / c_light);
            double exponential_real = std::cos(phase);
            double exponential_imag = std::sin(phase);
            double val_1 = vector[0] * exponential_real * denom * time_step * constant;
            double val_2 = vector[0] * exponential_imag * denom * time_step * constant;
            double val_3 = vector[1] * exponential_real * denom * time_step * constant;
            double val_4 = vector[1] * exponential_imag * denom * time_step * constant;
            double val_5 = vector[2] * exponential_real * denom * time_step * constant;
            double val_6 = vector[2] * exponential_imag * denom * time_step * constant;

            double integration_multiplier = 1;
            if (last_was_nan) {
              if (!std::isnan(val_1) || !std::isnan(val_2) || !std::isnan(val_3) || !std::isnan(val_4) || !std::isnan(val_5) || !std::isnan(val_6)) {
                last_was_nan = false;
                assert(!std::isnan(val_1) && !std::isnan(val_2) && !std::isnan(val_3) && !std::isnan(val_4) && !std::isnan(val_5) && !std::isnan(val_6));
                //integration_multiplier = 0.5;
              }
              else {
                continue;
              }
            }
            else {
              if (std::isnan(val_1) || std::isnan(val_2) || std::isnan(val_3) || std::isnan(val_4) || std::isnan(val_5) || std::isnan(val_6)) {
                break;
              }
              //assert(!std::isnan(val_1) && !std::isnan(val_2) && !std::isnan(val_3) && !std::isnan(val_4) && !std::isnan(val_5) && !std::isnan(val_6));
              //integration_multiplier = (o == n_steps) ? 0.5 : 1.0;
            }
            result[result_index + 0] += val_1 * integration_multiplier;
            result[result_index + 1] += val_2 * integration_multiplier;
            result[result_index + 2] += val_3 * integration_multiplier;
            result[result_index + 3] += val_4 * integration_multiplier;
            result[result_index + 4] += val_5 * integration_multiplier;
            result[result_index + 5] += val_6 * integration_multiplier;
          }
        }
      }
    }
  }
}

void compute_radiation_list_nan(double* __restrict__ result,
  Particle* __restrict__ data, EnergyPhixPhiy* __restrict__ inputs,
  std::size_t n_particles, std::size_t n_steps, std::size_t n_inputs,
  double time_step)
{
  SignalHandlerHelper signalhandlerhelper{};

  //std::size_t i = 0;
  //int percent = -1;

  for (std::size_t j = 0; j != n_inputs; ++j) {
    double frequency = inputs[j].energy / hbar_ev;
    double phi_x = inputs[j].phi_x;
    double phi_y = inputs[j].phi_y;

    std::size_t result_index = 6 * j;

    for (std::size_t a = 0; a != 6; ++a)
      result[result_index + a] = 0.0;

    for (std::size_t m = 0; m != n_particles; ++m) {
      bool last_was_nan = true;
      for (std::size_t o = 0; o != n_steps + 1; ++o) {
        /*
        int new_percent = static_cast<int>((100 * i) / (n_inputs * n_particles * (n_steps + 1)));
        if (percent != new_percent) {
          percent = new_percent;
          std::cout << percent << '%' << std::endl;
        }
        ++i;
        */

        double t = time_step * o;

        Particle particle = data[o + m * (n_steps + 1)];

        std::array<double, 3> n;
        std::array<double, 3> b;
        std::array<double, 3> bd;
        n[0] = std::sin(phi_x);
        n[1] = std::sin(phi_y);
        n[2] = std::sqrt(1 - std::pow(n[0], 2) - std::pow(n[1], 2));
        b[0] = particle.bx;
        b[1] = particle.by;
        b[2] = std::sqrt(1 - std::pow(particle.g, -2) - std::pow(particle.bx, 2) - std::pow(particle.by, 2));
        bd[0] = particle.bxd;
        bd[1] = particle.byd;
        bd[2] = -(particle.gd * std::pow(particle.g, -3) + particle.bx * particle.bxd + particle.by * particle.byd) / b[2];
        auto vector = cross(n, cross(subtract(n, b), bd));
        auto denom = std::pow(1 - dot(b, n), -2);
        double n_transverse2 = std::pow(n[0], 2) + std::pow(n[1], 2);
        double value = 0.5 * n_transverse2 + 0.125 * std::pow(n_transverse2, 2) + 0.0625 * std::pow(n_transverse2, 3);
        double phase = frequency * ((t * value) - (n[0] * particle.x + n[1] * particle.y + n[2] * particle.zeta) / c_light);
        double exponential_real = std::cos(phase);
        double exponential_imag = std::sin(phase);
        double val_1 = vector[0] * exponential_real * denom * time_step * constant;
        double val_2 = vector[0] * exponential_imag * denom * time_step * constant;
        double val_3 = vector[1] * exponential_real * denom * time_step * constant;
        double val_4 = vector[1] * exponential_imag * denom * time_step * constant;
        double val_5 = vector[2] * exponential_real * denom * time_step * constant;
        double val_6 = vector[2] * exponential_imag * denom * time_step * constant;

        double integration_multiplier;
        if (last_was_nan) {
          if (!std::isnan(val_1) || !std::isnan(val_2) || !std::isnan(val_3) || !std::isnan(val_4) || !std::isnan(val_5) || !std::isnan(val_6)) {
            last_was_nan = false;
            assert(!std::isnan(val_1) && !std::isnan(val_2) && !std::isnan(val_3) && !std::isnan(val_4) && !std::isnan(val_5) && !std::isnan(val_6));
            integration_multiplier = 0.5;
          }
          else {
            continue;
          }
        }
        else {
          assert(!std::isnan(val_1) && !std::isnan(val_2) && !std::isnan(val_3) && !std::isnan(val_4) && !std::isnan(val_5) && !std::isnan(val_6));
          integration_multiplier = (o == n_steps) ? 0.5 : 1.0;
        }
        result[result_index + 0] += val_1 * integration_multiplier;
        result[result_index + 1] += val_2 * integration_multiplier;
        result[result_index + 2] += val_3 * integration_multiplier;
        result[result_index + 3] += val_4 * integration_multiplier;
        result[result_index + 4] += val_5 * integration_multiplier;
        result[result_index + 5] += val_6 * integration_multiplier;
      }
    }
  }
}
