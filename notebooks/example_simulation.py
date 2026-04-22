"""
example_simulation.py
---------------------
Demonstrates a typical simulation workflow:
  1. Define physical parameters (based on Milul2023 / Sivak2023 experiments).
  2. Sweep over mean photon numbers.
  3. Run QSP-based Fock-state preparation via nice_sim_multiple.
  4. Compute fidelities and plot.

Run from the repository root:
    python notebooks/example_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt

from cqed import nice_sim_multiple, fid_giver, number_dist
from qutip import basis

# ---------------------------------------------------------------------------
# Physical parameters  (all rates in rad/ms to match chi units)
# ---------------------------------------------------------------------------
chi = 2 * np.pi * 41           # dispersive coupling  [rad/ms]  (Milul2023)
omega_con = chi * 200          # control Rabi freq    [rad/ms]
g_c = chi * 200                # displacement drive   [rad/ms]

gamma_c = 2 * np.pi * 6.4e-3  # cavity decay rate    (T1 ~ 25 ms)
gamma_qr = 2 * np.pi * 0.530  # qubit relaxation     (T1 ~ 300 µs)
gamma_qz = gamma_qr * 4.5     # qubit dephasing      (T2 ~ 60 µs)
n_crit = 1000                  # critical photon number

param_lib = {
    'chi': chi,
    'omega_con': omega_con,
    'gamma_c': gamma_c,
    'gamma_cdeph': 0.0,
    'gamma_qr': gamma_qr,
    'gamma_qz': gamma_qz,
    'n_crit': n_crit,
}

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
num_list = np.arange(18, 140, 32)   # target Fock numbers to prepare

# ---------------------------------------------------------------------------
# Run simulation: Wz basis, single measurement, cavity decay only
# ---------------------------------------------------------------------------
print("Running Wz-basis simulation (CavOnly noise)...")
states_Wz, _, succ_Wz = nice_sim_multiple(
    conv='Wz',
    num_list=num_list,
    output_list=[1],
    cancel='full_cancel',
    arg_list=['Diss', 'FockFromCoh', 'CavOnly'],
    custom_list=[],
    param_lib=param_lib,
)

# ---------------------------------------------------------------------------
# Compute fidelities
# ---------------------------------------------------------------------------
fid_Wz = fid_giver(states_Wz, num_list, basis(2, 0), 'Fock')

# Analytic estimate: fidelity ≈ exp(-π/chi * nbar * γ_c / 4)
fid_analytic = np.exp(-np.pi / chi * (num_list * gamma_c / 4))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(num_list, fid_Wz, 'o-', label='Wz (CavOnly)')
ax.plot(num_list, fid_analytic, '--', label='Analytic bound')
ax.set_xlabel('Target photon number $\\bar{n}$')
ax.set_ylabel('Fidelity')
ax.set_title('Fock-state fidelity vs. photon number')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(num_list, succ_Wz, 's-', label='Success probability')
ax.set_xlabel('Target photon number $\\bar{n}$')
ax.set_ylabel('Probability')
ax.set_title('QSP measurement success probability')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fock_prep_results.png', dpi=150)
plt.show()
print("Figure saved to fock_prep_results.png")
