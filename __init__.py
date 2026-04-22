"""
cqed
====
Quantum simulation library for QSP-based state preparation
in dispersively coupled qubit–cavity (cQED) systems.

Submodules
----------
measurement   : Measurement operators, photon-number distributions,
                fidelity and trace-distance metrics.
pulses        : QSP phase angles, time-domain pulse sequences,
                system Hamiltonians, and collapse operators.
simulation    : State preparation routines and high-level simulation drivers.
number_theory : CRT-based search for optimal QSP measurement parameters.
"""

from .measurement import (
    num_cond_succ,
    num_cond_fail,
    number_dist,
    number_dist_succ,
    number_dist_fail,
    proj_rk,
    r_legged_TrDist,
    fid_list,
    fid_giver,
    fid_of_just_cat,
    equal_sup,
)

from .pulses import (
    fake_phases_general,
    fake_phases,
    fake_phases_odd,
    Phase_to_time,
    Phase_to_time_Wz,
    square_pulses,
    square_pulses_Wz,
    pulse_data_had,
    pulse_data_had_Wz,
    pulse_data_shift,
    pulse_data_shift_Wz,
    pulse_data_phase,
    pulse_data_phase_Wz,
    pulse_data_signal_OFF,
    pulse_data_signal_OFF_Wz,
    H_full_reduced,
    H_full_reduced_Wz,
    C_OPS,
)

from .simulation import (
    prep_Coh_State,
    qsp_meas,
    multiple_meas,
    nice_sim_multiple,
    nice_sim_multiple_rdep,
    Fock_prep,
)

from .number_theory import (
    coprime,
    nth_prime,
    prime_factors,
    rk_find_best_prime,
)

__all__ = [
    # measurement
    'num_cond_succ', 'num_cond_fail',
    'number_dist', 'number_dist_succ', 'number_dist_fail',
    'proj_rk', 'r_legged_TrDist',
    'fid_list', 'fid_giver', 'fid_of_just_cat', 'equal_sup',
    # pulses
    'fake_phases_general', 'fake_phases', 'fake_phases_odd',
    'Phase_to_time', 'Phase_to_time_Wz',
    'square_pulses', 'square_pulses_Wz',
    'pulse_data_had', 'pulse_data_had_Wz',
    'pulse_data_shift', 'pulse_data_shift_Wz',
    'pulse_data_phase', 'pulse_data_phase_Wz',
    'pulse_data_signal_OFF', 'pulse_data_signal_OFF_Wz',
    'H_full_reduced', 'H_full_reduced_Wz', 'C_OPS',
    # simulation
    'prep_Coh_State', 'qsp_meas', 'multiple_meas',
    'nice_sim_multiple', 'nice_sim_multiple_rdep', 'Fock_prep',
    # number theory
    'coprime', 'nth_prime', 'prime_factors', 'rk_find_best_prime',
]
