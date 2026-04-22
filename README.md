# cQED-QSP: Quantum Signal Processing for State Preparation in cQED

A Python simulation library for **QSP-based photon-number state preparation** in dispersively coupled qubit–cavity (cQED) systems. The code simulates the full open-system dynamics of a qubit–cavity pair and implements a Quantum Signal Processing (QSP) measurement protocol that projects an input coherent state onto a target Fock state or multi-legged cat state.

This library accompanies the work described in [paper/preprint citation here].

---

## Physics Background

The system consists of a transmon qubit dispersively coupled to a microwave cavity. In the rotating frame, the Hamiltonian is:

$$H = -\chi \, \sigma_x \otimes \hat{n} \quad (\text{Wx basis})$$
$$H = -\chi \, \sigma_z \otimes \hat{n} \quad (\text{Wz basis})$$

where $\hat{n} = a^\dagger a$ is the cavity photon-number operator and $\chi$ is the dispersive coupling strength.

**The QSP protocol** applies a sequence of qubit rotations and free evolution intervals to engineer a polynomial function of $\hat{n}$ acting on the qubit. Measuring the qubit then projects the cavity onto a subspace of photon numbers — effectively implementing a modular number measurement. Repeating with two coprime moduli $r_1$, $r_2$ (chosen via the Chinese Remainder Theorem) localises the cavity state to a single Fock number.

**Two signal bases** are implemented:
- **Wx** (`conv='Wx'`): The cavity signal enters via $\sigma_x$. Requires Hadamard gate slots in the pulse sequence.
- **Wz** (`conv='Wz'`): The cavity signal enters via $\sigma_z$. No Hadamard gates required; generally lower overhead.

---

## Repository Structure

```
cqed_qsp/
├── cqed/                        # Main library package
│   ├── __init__.py              # Public API exports
│   ├── measurement.py           # Measurement operators, distributions, fidelities
│   ├── pulses.py                # QSP phases, pulse sequences, Hamiltonians
│   ├── simulation.py            # State prep and simulation drivers
│   └── number_theory.py        # CRT-based parameter search
│
├── notebooks/
│   └── example_simulation.py   # End-to-end example workflow
│
├── tests/
│   └── test_cqed.py            # Unit tests (pytest)
│
├── requirements.txt
└── README.md
```

---

## Installation

**Clone and install dependencies:**

```bash
git clone https://github.com/<your-username>/cqed-qsp.git
cd cqed-qsp
pip install -r requirements.txt
```

**Verify the install:**

```bash
pytest tests/
```

All tests should pass. The test suite covers number theory, phase generation, pulse logic, measurement operators, and collapse operators — without requiring a full simulation run (which can take minutes).

---

## Quick Start

```python
import numpy as np
from cqed import nice_sim_multiple, fid_giver
from qutip import basis

# Physical parameters (Milul2023 / Sivak2023 experiment)
chi       = 2 * np.pi * 41          # dispersive coupling [rad/ms]
omega_con = chi * 200               # control Rabi frequency
gamma_c   = 2 * np.pi * 6.4e-3     # cavity decay (T1 ~ 25 ms)
gamma_qr  = 2 * np.pi * 0.530      # qubit relaxation (T1 ~ 300 µs)
gamma_qz  = gamma_qr * 4.5         # qubit dephasing (T2 ~ 60 µs)

param_lib = {
    'chi': chi, 'omega_con': omega_con,
    'gamma_c': gamma_c, 'gamma_cdeph': 0.0,
    'gamma_qr': gamma_qr, 'gamma_qz': gamma_qz,
    'n_crit': 1000,
}

# Sweep over target Fock numbers
num_list = np.arange(18, 140, 32)

# Run Wz-basis simulation with cavity decay only
states, _, succ_probs = nice_sim_multiple(
    conv       = 'Wz',
    num_list   = num_list,
    output_list= [1],                      # one measurement, outcome = failure
    cancel     = 'full_cancel',
    arg_list   = ['Diss', 'FockFromCoh', 'CavOnly'],
    custom_list= [],
    param_lib  = param_lib,
)

# Compute fidelity with the target Fock state
fidelities = fid_giver(states, num_list, basis(2, 0), 'Fock')
print("Fidelities:", fidelities)
```

---

## Key API

### `nice_sim_multiple(conv, num_list, output_list, cancel, arg_list, custom_list, param_lib)`

The main simulation driver. Sweeps over photon numbers and runs repeated QSP measurements.

| Argument | Type | Description |
|---|---|---|
| `conv` | `str` | Signal basis: `'Wx'` or `'Wz'` |
| `num_list` | `array` | Target photon numbers $\bar{n}$ |
| `output_list` | `list[int]` | Qubit outcomes per round (0=success, 1=failure) |
| `cancel` | `str` | `'full_cancel'`, `'cancel_SQ'`, or `'half_cancel'` |
| `arg_list` | `list` | `[diss_mode, state_type, noise_model]` |
| `param_lib` | `dict` | Physical parameters |

**`arg_list` options:**

- `diss_mode`: `'Diss'` (open system) or `'NoDiss'` (unitary)
- `state_type`: `'FockFromCoh'`, `'Fock'`, `'Cat'`, `'CustomFock'`, `'Fock_2_meas'`
- `noise_model`: `'CavOnly'`, `'NoQDeph'`, `'QDecOnly'`, `'QDephOnly'`, `'Full'`

### `fid_giver(state_list, num_list, meas_out, state_type)`

Compute the fidelity of each output state with the ideal target state.

- `meas_out`: qubit part of the target — `basis(2,0)` or `basis(2,1)`
- `state_type`: `'Fock'` or `'Cat'`

### `rk_find_best_prime(nbar, m)`

Find the optimal CRT moduli $(r_1, r_2)$ for a two-round Fock-state preparation targeting $|n_\text{bar}\rangle$. Returns `(r1*r2, [r1, r2], [k1, k2])` or `'NA'` if no pair satisfies the constraints.

### `C_OPS(gamma_c, gamma_cdeph, gamma_qr, gamma_qz, n_crit, max_ph)`

Build the collapse-operator list for a given noise model.

---

## Known Issues and Limitations

The following issues were identified and fixed in the refactored code:

1. **`flip_meas_q` undefined in `r_legged_Cat`** — the `follow='follow'` branch of the original function used `flip_meas_q = tensor(sigmax(), qeye(max_ph))` without defining it. This is corrected in the refactored version by defining the operator at the top of the function.

2. **`r_legged_Cat_sweep` used `exec()` and undefined globals** — `nbar_var`, `g_c`, `chi`, `omega_con`, and `gamma_cdeph` were referenced inside a loop without being in scope. The function also used `exec(f"{which} = {var}")` which does not actually set local variables in Python 3. This function requires a design decision from the author (what exactly is being swept?) before it can be safely included; it has been omitted from this refactor.

3. **`fid_list_DensityMat`** — marked in the original code with the comment *"I am not sure if this function actually is good for anything"*. Omitted as a duplicate of `fid_list`.

4. **`nice_sim_multiple_Wz_test` and `nice_sim_multiple_test`** — defined inside the notebook as temporary experiments. Consolidated into the general `nice_sim_multiple` with the `conv` parameter.

5. **Scattered `print()` debug statements** — replaced with structured comments. Re-add with Python's `logging` module if runtime feedback is needed.

6. **Large blocks of commented-out code** — removed. The old phase functions (`fake_phases` pre–May 2024) are preserved but clearly labelled as legacy in the docstrings.

7. **`Kerr` printed at every call** of `H_full_reduced_Wz` — removed the `print` statement; the Kerr value is documented in the docstring.

---

## Physical Parameters (Reference Values)

Based on Milul et al. (2023) and Sivak et al. (2022/2023):

| Parameter | Symbol | Value | Reference |
|---|---|---|---|
| Dispersive coupling | $\chi$ | $2\pi \times 41$ kHz | Milul2023 |
| Cavity lifetime | $T_{1,c}$ | 25 ms | Milul2023 |
| Qubit $T_1$ | $T_{1,q}$ | ~300 µs | Sivak2022 |
| Qubit $T_2$ | $T_{2,q}$ | ~60 µs | Sivak2022 |
| Critical photon number | $n_\text{crit}$ | 1000 | — |

---

## Running Tests

```bash
pytest tests/ -v
```

The tests check:
- Number-theoretic utilities (coprimality, CRT pair search)
- QSP phase sum constraints (phases must sum to $\pi/2$)
- Time-list monotonicity
- Pulse channel logic (Wz Hadamard channel always zero)
- Measurement operator completeness relations
- Projector idempotency ($P^2 = P$)
- Collapse operator dimensions

---

## Citation

If you use this code, please cite:

```
[Your paper citation here]
```

## License

[Your license here]
