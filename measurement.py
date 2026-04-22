"""
measurement.py
--------------
Measurement operators and conditional probability distributions for
a qubit–cavity system (qubit ⊗ Fock-space).

The Hilbert space convention throughout is:
    |qubit⟩ ⊗ |photon number⟩

Qubit basis:  |0⟩ = success/|g⟩,  |1⟩ = failure/|e⟩
"""

import numpy as np
from qutip import (
    basis, tensor, qeye, expect, fidelity, tracedist,
    sigmax, sigmaz, sigmap, sigmam, destroy, coherent,
)


# ---------------------------------------------------------------------------
# Elementary conditional projectors
# ---------------------------------------------------------------------------

def num_cond_succ(j, max_ph):
    """Projector onto qubit=|0⟩ (success) AND cavity photon number = j."""
    return tensor(
        basis(2, 0) * basis(2, 0).dag(),
        basis(max_ph, j) * basis(max_ph, j).dag(),
    )


def num_cond_fail(j, max_ph):
    """Projector onto qubit=|1⟩ (failure) AND cavity photon number = j."""
    return tensor(
        basis(2, 1) * basis(2, 1).dag(),
        basis(max_ph, j) * basis(max_ph, j).dag(),
    )


# ---------------------------------------------------------------------------
# Photon-number distributions
# ---------------------------------------------------------------------------

def number_dist(state, max_ph):
    """
    Marginal photon-number distribution (summed over qubit).

    Parameters
    ----------
    state : Qobj
        Joint qubit–cavity state (ket or density matrix).
    max_ph : int
        Cavity Hilbert-space dimension.

    Returns
    -------
    ndarray of length max_ph
        P(n) for n = 0, …, max_ph-1.
    """
    dist = []
    for j in np.arange(max_ph):
        meas = tensor(qeye(2), basis(max_ph, j) * basis(max_ph, j).dag())
        dist = np.append(dist, expect(meas, state))
    return dist


def number_dist_succ(state, max_ph):
    """
    Photon-number distribution conditioned on qubit = |0⟩ (success).

    Returns
    -------
    ndarray of length max_ph
        P(n | success).
    """
    succ_op = tensor(basis(2, 0) * basis(2, 0).dag(), qeye(max_ph))
    succ_prob = expect(succ_op, state)
    dist = []
    for j in np.arange(max_ph):
        num_j = expect(num_cond_succ(j, max_ph), state)
        dist = np.append(dist, num_j / succ_prob)
    return dist


def number_dist_fail(state, max_ph):
    """
    Photon-number distribution conditioned on qubit = |1⟩ (failure).

    Returns
    -------
    ndarray of length max_ph
        P(n | failure).
    """
    fail_op = tensor(basis(2, 1) * basis(2, 1).dag(), qeye(max_ph))
    fail_prob = expect(fail_op, state)
    dist = []
    for j in np.arange(max_ph):
        num_j = expect(num_cond_fail(j, max_ph), state)
        dist = np.append(dist, num_j / fail_prob)
    return dist


# ---------------------------------------------------------------------------
# r-legged cat-state projectors and fidelity metrics
# ---------------------------------------------------------------------------

def proj_rk(r, k, max_ph):
    """
    Projector onto the subspace spanned by Fock states |n⟩ with n ≡ k (mod r),
    tensored with qubit |0⟩.

    Parameters
    ----------
    r : int
        Number of legs (periodicity).
    k : int
        Residue class (0 ≤ k < r).
    max_ph : int
        Cavity Hilbert-space dimension.
    """
    first_state = tensor(basis(2, 0), basis(max_ph, k))
    meas_target = first_state * first_state.dag()
    for j in np.arange(max_ph):
        if j != k and j % r == k:
            new_state = tensor(basis(2, 0), basis(max_ph, j))
            meas_target = meas_target + new_state * new_state.dag()
    return meas_target


def r_legged_TrDist(psi, psi_0, r, k, max_ph, args):
    """
    Trace distance between a state and the ideal r-legged cat state
    obtained by projecting psi_0 with proj_rk(r, k).

    Parameters
    ----------
    psi : Qobj
        State to compare (will be normalised internally).
    psi_0 : Qobj
        Reference state before projection (e.g. initial coherent state).
    r, k : int
        Cat-state parameters.
    max_ph : int
        Cavity Hilbert-space dimension.
    args : str
        'Diss' for open-system states (density matrices),
        anything else for pure states (kets).
    """
    psi = psi / psi.norm()
    psi_0 = psi_0 / psi_0.norm()
    meas_target = proj_rk(r, k, max_ph)
    if args == 'Diss':
        psi_target = meas_target * psi_0 * meas_target.dag()
    else:
        psi_target = meas_target * psi_0
    psi_target = psi_target / psi_target.norm()
    return tracedist(psi, psi_target)


# ---------------------------------------------------------------------------
# Fidelity lists
# ---------------------------------------------------------------------------

def fid_list(psi_list, num_list):
    """
    Compute the fidelity of each state in psi_list with the ideal
    sqrt(nbar)-legged cat state projected from a coherent state.

    Parameters
    ----------
    psi_list : list of Qobj
        Output states (kets or density matrices).
    num_list : array-like
        Mean photon numbers corresponding to each state.

    Returns
    -------
    fid_arr : ndarray
    perf_list : list of Qobj
        The ideal target density matrices used for comparison.
    """
    fid_arr = []
    perf_list = []
    for j in np.arange(len(psi_list)):
        nbar = num_list[j]
        r = np.ceil(np.sqrt(nbar))
        max_ph_target = nbar + 100
        perf_state = tensor(
            basis(2, 0),
            coherent(max_ph_target, np.sqrt(nbar), offset=0, method='operator'),
        )
        proj = proj_rk(r, 0, max_ph_target)
        perf_state = proj * perf_state
        perf_state = perf_state / perf_state.norm()
        perf_state = perf_state * perf_state.dag()
        perf_list.append(perf_state)
        fid_arr = np.append(fid_arr, fidelity(psi_list[j], perf_state))
    return fid_arr, perf_list


def fid_giver(state_list, num_list, meas_out, state_type):
    """
    Fidelity of each output state with the ideal target state,
    for either Fock or cat target states.

    Parameters
    ----------
    state_list : list of Qobj
    num_list : array-like
    meas_out : Qobj
        Qubit component of the target: basis(2,0) or basis(2,1).
    state_type : str
        'Fock' – target is |meas_out⟩ ⊗ |nbar⟩.
        'Cat'  – target is the projected cat state.

    Returns
    -------
    ndarray of fidelities.
    """
    fid_arr = []
    for ind in np.arange(len(num_list)):
        num = num_list[ind]
        max_ph_var = int(num + 100)
        if state_type == 'Fock':
            meas_st = tensor(meas_out, basis(max_ph_var, num))
        elif state_type == 'Cat':
            r = np.ceil(np.sqrt(num))
            meas_st = tensor(
                meas_out,
                coherent(max_ph_var, np.sqrt(num), offset=0, method='operator'),
            )
            proj = proj_rk(r, 0, max_ph_var)
            if meas_out == basis(2, 1):
                SigmaX = tensor(sigmax(), qeye(max_ph_var))
                meas_st = SigmaX * proj * SigmaX * meas_st
            elif meas_out == basis(2, 0):
                meas_st = proj * meas_st
            meas_st = meas_st / meas_st.norm()
        fid_arr = np.append(fid_arr, fidelity(meas_st, state_list[ind]))
    return fid_arr


def fid_of_just_cat(num_list, chi, gamma_cav):
    """
    Fidelity of a cat state that has evolved under cavity decay only
    (no measurements), used as a reference baseline.

    Parameters
    ----------
    num_list : array-like
        Mean photon numbers to sweep.
    chi : float
        Dispersive coupling strength (rad/s or consistent units).
    gamma_cav : float
        Cavity decay rate.

    Returns
    -------
    ndarray of fidelities.
    """
    fid_arr = []
    for nbar_var in num_list:
        max_ph_var = nbar_var + 100
        a = tensor(qeye(2), destroy(max_ph_var))
        r = np.ceil(np.sqrt(nbar_var))
        proj = proj_rk(r, 0, max_ph_var)
        psi_init = tensor(
            basis(2, 0),
            coherent(max_ph_var, np.sqrt(nbar_var), offset=0, method='operator'),
        )
        psi_init = proj * psi_init
        psi_init = psi_init / psi_init.norm()
        time_prep = np.linspace(0, np.pi / chi, 1000)
        from qutip import mesolve
        result = mesolve(
            0.001 * tensor(qeye(2), qeye(max_ph_var)),
            psi_init,
            time_prep,
            [np.sqrt(gamma_cav) * a],
        )
        psi_final = result.states[-1]
        psi_final = psi_final / psi_final.norm()
        fid_arr = np.append(fid_arr, fidelity(psi_init, psi_final))
    return fid_arr


def equal_sup(max_ph):
    """
    Equal superposition of all Fock states |0⟩…|max_ph-1⟩,
    tensored with qubit |0⟩, returned as a density matrix.
    """
    psi_init = tensor(basis(2, 0), basis(max_ph, 0))
    for j in np.arange(max_ph - 1):
        psi_init = psi_init + tensor(basis(2, 0), basis(max_ph, j + 1))
    psi_init = (1 / np.sqrt(max_ph)) * psi_init
    return psi_init * psi_init.dag()
