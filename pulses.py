"""
pulses.py
---------
Quantum Signal Processing (QSP) phase angles, time-domain pulse sequences,
and system Hamiltonians for a dispersively coupled qubit–cavity system.

Two signal bases are supported:
    'Wx' – signal via σ_x coupling  (uses Hadamard gates in the sequence)
    'Wz' – signal via σ_z coupling  (no Hadamard gates needed)
"""

import numpy as np
from qutip import (
    basis, tensor, qeye, destroy,
    sigmax, sigmay, sigmaz, sigmap, sigmam,
    mesolve,
)


# ---------------------------------------------------------------------------
# QSP phase angle generation
# ---------------------------------------------------------------------------

def _qsp_cosine_phases(r_var, shift):
    """
    Internal helper: generate the interior QSP phases using the cosine
    approximation kernel func(), for a given number of legs r_var and
    an optional shift parameter.
    """
    x_disc = np.arange(r_var) - (r_var + 1) / 2
    a = b = c = 1.0 / (r_var + shift)
    return a / np.pi * 4 + b / np.pi * 2 * np.cos(4 * c * (x_disc + 1))


def fake_phases_general(r):
    """
    General QSP phase sequence for an r-legged cat projection.

    Works for both even and odd r by rounding up to the next even number
    for the interior phases and then enforcing sum = π/2 via cap phases.

    Parameters
    ----------
    r : int or float
        Number of legs.

    Returns
    -------
    ndarray
        Phase sequence of length 2*ceil(r/2) + 1.
    """
    new_r = 2 * np.ceil(r / 2)
    x_disc = np.arange(new_r - 1) - new_r / 2
    a = b = c = 1.0 / r
    interior = a / np.pi * 4 + b / np.pi * 2 * np.cos(4 * c * (x_disc + 1))
    cap_phase = (np.pi / 2 - interior.sum()) / 2
    return np.append(cap_phase, np.append(interior, cap_phase))


def fake_phases(r):
    """Even-r QSP phase sequence (legacy; prefer fake_phases_general)."""
    r_var = r - 1
    interior = _qsp_cosine_phases(r_var, 1.0)
    cap_phase = (np.pi / 2 - interior.sum()) / 2
    return np.append(cap_phase, np.append(interior, cap_phase))


def fake_phases_odd(r):
    """Odd-r QSP phase sequence (legacy; prefer fake_phases_general)."""
    r_var = r - 1
    interior = _qsp_cosine_phases(r_var, 0.0)
    cap_phase = (np.pi / 2 - interior.sum()) / 2
    return np.append(cap_phase, np.append(interior, cap_phase))


# ---------------------------------------------------------------------------
# Phase angles → time markers
# ---------------------------------------------------------------------------

def Phase_to_time(chi, omega_con, r, r_tilde, k):
    """
    Convert QSP phases to a list of time markers for the Wx-basis
    pulse sequence (includes Hadamard-gate time slots).

    A small random phase noise (0.01 relative) is applied to model
    realistic control imperfections.

    Parameters
    ----------
    chi : float
        Dispersive coupling (rad / time-unit).
    omega_con : float
        Control drive Rabi frequency.
    r : int
        Number of legs (determines phase count).
    r_tilde : int
        Unused legacy parameter kept for API compatibility.
    k : int
        Photon-number offset (shift).

    Returns
    -------
    ndarray
        Ordered time markers for all pulse edges.
    """
    n_phases = int(2 * np.ceil(r / 2) + 1)
    rand_phase_fac = 1 + np.random.normal(0, 0.01, n_phases)
    QSP_Phases = fake_phases_general(r) * rand_phase_fac

    signal_break = np.pi / (r * chi)
    shift_break = (np.pi / (r * omega_con)) * (-k)

    times_list = [0]
    # Left Hadamard slot
    time_tot = np.pi / 4 / omega_con
    for j in np.arange(len(QSP_Phases) - 1):
        QSP_phase_time = time_tot + np.abs(QSP_Phases[j]) / omega_con
        QSP_phase_signal = QSP_phase_time + signal_break
        QSP_phase_signal_shift = QSP_phase_signal + np.abs(shift_break)
        times_list = np.append(
            times_list,
            [time_tot, QSP_phase_time, QSP_phase_signal],
        )
        time_tot = QSP_phase_signal_shift

    times_list = np.append(
        times_list,
        [time_tot, time_tot + QSP_Phases[-1] / omega_con],
    )
    time_tot = time_tot + QSP_Phases[-1] / omega_con
    # Right Hadamard slot
    times_list = np.append(times_list, [time_tot + np.pi / 4 / omega_con])
    return times_list


def Phase_to_time_Wz(chi, omega_con, r, r_tilde, k):
    """
    Convert QSP phases to time markers for the Wz-basis pulse sequence
    (no Hadamard gates; qubit rotations along z).

    Parameters are the same as Phase_to_time.
    """
    n_phases = int(2 * np.ceil(r / 2) + 1)
    rand_phase_fac = 1 + np.random.normal(0, 0.01, n_phases)
    QSP_Phases = fake_phases_general(r) * rand_phase_fac

    signal_break = np.pi / (r * chi)
    shift_break = (np.pi / (r * omega_con)) * (-k)

    times_list = [0]
    time_tot = 0.0
    for j in np.arange(len(QSP_Phases) - 1):
        QSP_phase_time = time_tot + np.abs(QSP_Phases[j]) / omega_con
        QSP_phase_signal = QSP_phase_time + signal_break
        QSP_phase_signal_shift = QSP_phase_signal + np.abs(shift_break)
        times_list = np.append(
            times_list,
            [time_tot, QSP_phase_time, QSP_phase_signal],
        )
        time_tot = QSP_phase_signal_shift

    times_list = np.append(
        times_list,
        [time_tot, time_tot + QSP_Phases[-1] / omega_con],
    )
    time_tot = time_tot + QSP_Phases[-1] / omega_con
    times_list = np.append(times_list, [time_tot])
    return times_list


# ---------------------------------------------------------------------------
# Square-pulse logic
# ---------------------------------------------------------------------------

def _calc_order(time_list, t_cont):
    """
    Return the index of the time interval that contains t_cont.
    Specifically, the number of time_list entries that are ≤ t_cont.
    """
    counter = 0
    for j in np.arange(len(time_list)):
        if j < len(time_list) - 1 and t_cont >= time_list[j]:
            counter += 1
        else:
            return counter
    return counter  # fallback


def square_pulses(time_list, t_cont):
    """
    Return the on/off pattern [had, signal, shift, phase] at time t_cont
    for the Wx-basis sequence.

    Values are in {-1, 0, 1}; -1 means 'cancel' (active with opposite sign).
    """
    bin_num = _calc_order(time_list, t_cont)
    remain = np.mod(bin_num - 1, 3)

    if bin_num == 1:
        had, signal, shift, phase = -1, 0, 0, 0
    elif bin_num == len(time_list) - 1:
        had, signal, shift, phase = 1, 0, 0, 0
    elif remain == 0:
        had, signal, shift, phase = 0, 0, 1, 0
    elif remain == 1:
        had, signal, shift, phase = 0, 0, 0, -1
    elif remain == 2:
        had, signal, shift, phase = 0, -1, 0, 0
    else:
        had, signal, shift, phase = 0, 0, 0, 0

    return np.array([had, signal, shift, phase])


def square_pulses_Wz(time_list, t_cont):
    """
    Return the on/off pattern [0, 0, shift, phase] at time t_cont
    for the Wz-basis sequence (had channel always 0).
    """
    bin_num = _calc_order(time_list, t_cont)
    remain = np.mod(bin_num - 1, 3)

    if remain == 0:
        signal, shift, phase = 0, 1, 0
    elif remain == 1:
        signal, shift, phase = 0, 0, -1
    elif remain == 2:
        signal, shift, phase = -1, 0, 0
    else:
        signal, shift, phase = 0, 0, 0

    return np.array([0, 0, shift, phase])


# ---------------------------------------------------------------------------
# Time-dependent pulse functions (QuTiP callback format)
# ---------------------------------------------------------------------------

def pulse_data_had(t, args):
    return square_pulses(args['time_list'], t)[0]

def pulse_data_had_Wz(t, args):
    return square_pulses_Wz(args['time_list'], t)[0]

def pulse_data_shift(t, args):
    return square_pulses(args['time_list'], t)[2]

def pulse_data_shift_Wz(t, args):
    return square_pulses_Wz(args['time_list'], t)[2]

def pulse_data_phase(t, args):
    return square_pulses(args['time_list'], t)[3]

def pulse_data_phase_Wz(t, args):
    return square_pulses_Wz(args['time_list'], t)[3]

def pulse_data_signal_OFF(t, args):
    """Cancel signal whenever phase, Hadamard, or shift is active (Wx basis)."""
    f = square_pulses(args['time_list'], t)
    return -(np.abs(f[3]) + np.abs(f[0]) + np.abs(f[2]))

def pulse_data_signal_OFF_Wz(t, args):
    """Cancel signal whenever phase or shift is active (Wz basis)."""
    f = square_pulses_Wz(args['time_list'], t)
    return -(np.abs(f[3]) + np.abs(f[0]) + np.abs(f[2]))


# ---------------------------------------------------------------------------
# Hamiltonians
# ---------------------------------------------------------------------------

def H_full_reduced(chi, omega_con, max_ph, nbar, cancel, coupling_type):
    """
    Build the time-dependent Hamiltonian list for the Wx-basis QSP sequence.

    Parameters
    ----------
    chi : float
        Dispersive (or Jaynes-Cummings) coupling strength.
    omega_con : float
        Control drive amplitude.
    max_ph : int
        Cavity Hilbert-space dimension.
    nbar : float
        Mean photon number (used for single-quadrature cancellation).
    cancel : str
        'cancel_SQ'  – cancel the single-quadrature (coherent) shift only.
        'full_cancel' – cancel the full coupling Hamiltonian.
        'half_cancel' – cancel half the coupling.
    coupling_type : str
        'Dispersive'  – H_coupling = chi * σ_x ⊗ n̂.
        anything else – Jaynes-Cummings: chi * (σ+ ⊗ a + σ- ⊗ a†).

    Returns
    -------
    list
        QuTiP time-dependent Hamiltonian list [H_0, [H_1, f_1(t)], …].
    """
    a = destroy(max_ph)
    H_control_had = omega_con * tensor(sigmay(), qeye(max_ph))
    H_control = omega_con * tensor(sigmaz(), qeye(max_ph))
    H_shift = omega_con * tensor(sigmax(), qeye(max_ph))

    if coupling_type == 'Dispersive':
        H_coupling = chi * tensor(sigmax(), a.dag() * a)
        if cancel == 'cancel_SQ':
            return [
                -H_coupling,
                [-chi * nbar * tensor(sigmax(), qeye(max_ph)), pulse_data_signal_OFF],
                [H_control, pulse_data_phase],
                [H_shift, pulse_data_shift],
                [H_control_had, pulse_data_had],
            ]
        elif cancel == 'full_cancel':
            return [
                -H_coupling,
                [-H_coupling, pulse_data_signal_OFF],
                [H_control, pulse_data_phase],
                [H_shift, pulse_data_shift],
                [H_control_had, pulse_data_had],
            ]
        elif cancel == 'half_cancel':
            return [
                -H_coupling,
                [-0.5 * H_coupling, pulse_data_signal_OFF],
                [H_control, pulse_data_phase],
                [H_shift, pulse_data_shift],
                [H_control_had, pulse_data_had],
            ]
        else:
            raise ValueError(f"Unknown cancel mode: '{cancel}'")
    else:
        # Jaynes-Cummings coupling
        H_coupling = chi * (tensor(sigmap(), a) + tensor(sigmam(), a.dag()))
        return [
            -H_coupling,
            [-H_coupling, pulse_data_signal_OFF],
            [H_control, pulse_data_phase],
            [H_shift, pulse_data_shift],
            [H_control_had, pulse_data_had],
        ]


def H_full_reduced_Wz(chi, omega_con, max_ph, nbar, cancel, coupling_type):
    """
    Build the time-dependent Hamiltonian list for the Wz-basis QSP sequence.

    Includes an optional Kerr nonlinearity term (currently hard-coded to
    chi/20000; set to 0 to disable).

    Parameters
    ----------
    (same as H_full_reduced, except coupling_type is currently unused
     as only the dispersive model is implemented for Wz)

    Returns
    -------
    list
        QuTiP time-dependent Hamiltonian list.
    """
    a = destroy(max_ph)
    H_control = omega_con * tensor(sigmax(), qeye(max_ph))
    H_shift = omega_con * tensor(sigmaz(), qeye(max_ph))

    # Dispersive + Kerr
    Kerr = -chi / 20000
    H_coupling_disp = chi * tensor(sigmaz(), a.dag() * a)
    H_coupling_kerr = (Kerr / 4) * tensor(
        sigmaz(), (a.dag() * a * a.dag() * a - nbar**2)
    )
    H_coupling = H_coupling_disp + H_coupling_kerr

    if cancel == 'cancel_SQ':
        shift_cancel = -(chi * nbar + Kerr / 4 * nbar**2) * tensor(
            sigmaz(), qeye(max_ph)
        )
        return [
            -H_coupling,
            [shift_cancel, pulse_data_signal_OFF_Wz],
            [H_control, pulse_data_phase_Wz],
            [H_shift, pulse_data_shift_Wz],
        ]
    elif cancel == 'full_cancel':
        return [
            -H_coupling,
            [-H_coupling, pulse_data_signal_OFF_Wz],
            [H_control, pulse_data_phase_Wz],
            [H_shift, pulse_data_shift_Wz],
        ]
    else:
        raise ValueError(f"Unknown cancel mode: '{cancel}'")


def C_OPS(gamma_c, gamma_cdeph, gamma_qr, gamma_qz, n_crit, max_ph):
    """
    Collapse operators for the open qubit–cavity system.

    Channels included:
        • Cavity photon loss:       √γ_c · (I ⊗ a)
        • Qubit relaxation:         √γ_qr · (σ- ⊗ I)
        • Qubit dephasing:          √(γ_qz/2) · (σ_z ⊗ I)
        • Dispersive qubit dephasing via photon number fluctuations
          (γ_qdeph_eff = γ_qz / (6 n_crit)):
              γ_qdeph_eff · (σ+ ⊗ a)   and   γ_qdeph_eff · (σ- ⊗ a†)

    Parameters
    ----------
    gamma_c : float
        Cavity decay rate.
    gamma_cdeph : float
        Cavity dephasing rate (currently unused / set to 0 in practice).
    gamma_qr : float
        Qubit relaxation rate.
    gamma_qz : float
        Qubit pure-dephasing rate.
    n_crit : float
        Critical photon number n_crit = (Δ/2g)².
    max_ph : int
        Cavity Hilbert-space dimension.

    Returns
    -------
    list of Qobj
        Collapse operator list ready for mesolve / mcsolve.
    """
    gamma_qdeph_eff = gamma_qz / (6 * n_crit)
    a = destroy(max_ph)
    disp_op1 = tensor(sigmap(), a)
    disp_op2 = tensor(sigmam(), a.dag())
    return [
        np.sqrt(gamma_c) * tensor(qeye(2), a),
        np.sqrt(gamma_qr) * tensor(destroy(2), qeye(max_ph)),
        np.sqrt(gamma_qz / 2) * tensor(sigmaz(), qeye(max_ph)),
        gamma_qdeph_eff * disp_op1,
        gamma_qdeph_eff * disp_op2,
    ]
