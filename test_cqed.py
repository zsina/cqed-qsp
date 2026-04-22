"""
tests/test_cqed.py
------------------
Unit tests for the cqed package.

Run with:  pytest tests/
"""

import numpy as np
import pytest
from qutip import basis, tensor, qeye, expect

from cqed import (
    num_cond_succ, num_cond_fail,
    number_dist, number_dist_succ, number_dist_fail,
    proj_rk,
    fake_phases_general, fake_phases, fake_phases_odd,
    Phase_to_time, Phase_to_time_Wz,
    square_pulses, square_pulses_Wz,
    C_OPS,
    coprime, nth_prime, prime_factors, rk_find_best_prime,
)


# ---------------------------------------------------------------------------
# Number theory
# ---------------------------------------------------------------------------

class TestNumberTheory:
    def test_coprime_basic(self):
        assert coprime(3, 5) is True
        assert coprime(4, 6) is False
        assert coprime(1, 100) is True

    def test_nth_prime(self):
        assert nth_prime(1) == 2
        assert nth_prime(2) == 3
        assert nth_prime(5) == 11

    def test_prime_factors_product_exceeds_nbar(self):
        _, product = prime_factors(30)
        assert product >= 30  # product meets or exceeds nbar

    def test_rk_find_best_prime_returns_valid_pair(self):
        result = rk_find_best_prime(50, 3)
        assert result != 'NA', "Should find a valid CRT pair for nbar=50"
        product, factors, residues = result
        r1, r2 = int(factors[0]), int(factors[1])
        k1, k2 = int(residues[0]), int(residues[1])
        assert coprime(r1, r2), "Returned factors must be coprime"
        assert r1 * r2 == product
        assert k1 == 50 % r1
        assert k2 == 50 % r2

    def test_rk_find_best_prime_discrimination_gap(self):
        m = 3
        result = rk_find_best_prime(50, m)
        assert result != 'NA'
        _, factors, _ = result
        # Product must exceed sigma = 6*sqrt(50)
        sigma = 6 * np.sqrt(50)
        assert factors[0] * factors[1] > sigma


# ---------------------------------------------------------------------------
# QSP phase generation
# ---------------------------------------------------------------------------

class TestPhaseGeneration:
    @pytest.mark.parametrize("r", [2, 3, 4, 5, 6, 7, 8])
    def test_phases_sum_to_half_pi(self, r):
        phases = fake_phases_general(r)
        assert abs(phases.sum() - np.pi / 2) < 1e-10, \
            f"Phases for r={r} should sum to π/2, got {phases.sum()}"

    @pytest.mark.parametrize("r", [2, 4, 6])
    def test_fake_phases_even(self, r):
        phases = fake_phases(r)
        assert abs(phases.sum() - np.pi / 2) < 1e-10

    @pytest.mark.parametrize("r", [3, 5, 7])
    def test_fake_phases_odd(self, r):
        phases = fake_phases_odd(r)
        assert abs(phases.sum() - np.pi / 2) < 1e-10

    def test_phase_count_general(self):
        for r in [3, 4, 5, 6]:
            phases = fake_phases_general(r)
            expected_len = int(2 * np.ceil(r / 2) + 1)
            assert len(phases) == expected_len, \
                f"r={r}: expected {expected_len} phases, got {len(phases)}"


# ---------------------------------------------------------------------------
# Time-list generation
# ---------------------------------------------------------------------------

class TestTimeList:
    def test_phase_to_time_is_sorted(self):
        chi = 2 * np.pi * 0.041
        omega_con = chi * 200
        times = Phase_to_time(chi, omega_con, 4, 4, 0)
        assert np.all(np.diff(times) >= 0), "Time list must be non-decreasing"

    def test_phase_to_time_wz_is_sorted(self):
        chi = 2 * np.pi * 0.041
        omega_con = chi * 200
        times = Phase_to_time_Wz(chi, omega_con, 4, 4, 0)
        assert np.all(np.diff(times) >= 0), "Wz time list must be non-decreasing"

    def test_time_list_starts_at_zero(self):
        chi = 2 * np.pi * 0.041
        omega_con = chi * 200
        times = Phase_to_time(chi, omega_con, 3, 3, 1)
        assert times[0] == 0.0


# ---------------------------------------------------------------------------
# Pulse logic
# ---------------------------------------------------------------------------

class TestPulseLogic:
    def test_square_pulses_returns_four_values(self):
        time_list = np.linspace(0, 1, 20)
        result = square_pulses(time_list, 0.5)
        assert len(result) == 4

    def test_square_pulses_wz_first_entry_zero(self):
        """Wz pulses should never activate the Hadamard channel."""
        time_list = np.linspace(0, 1, 20)
        for t in np.linspace(0.01, 0.99, 20):
            result = square_pulses_Wz(time_list, t)
            assert result[0] == 0, "Wz had channel must always be 0"

    def test_square_pulses_boundary(self):
        """At t=0 (first interval), Hadamard should be active."""
        time_list = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        result = square_pulses(time_list, 0.05)
        assert result[0] != 0, "Hadamard should be active in first interval"


# ---------------------------------------------------------------------------
# Measurement operators
# ---------------------------------------------------------------------------

class TestMeasurementOperators:
    def test_num_cond_succ_trace(self):
        max_ph = 5
        total = sum(num_cond_succ(j, max_ph) for j in range(max_ph))
        succ_id = tensor(basis(2, 0) * basis(2, 0).dag(), qeye(max_ph))
        diff = (total - succ_id).norm()
        assert diff < 1e-10, "Sum of succ projectors must equal success POVM"

    def test_num_cond_fail_trace(self):
        max_ph = 5
        total = sum(num_cond_fail(j, max_ph) for j in range(max_ph))
        fail_id = tensor(basis(2, 1) * basis(2, 1).dag(), qeye(max_ph))
        diff = (total - fail_id).norm()
        assert diff < 1e-10, "Sum of fail projectors must equal failure POVM"

    def test_number_dist_normalised(self):
        max_ph = 10
        # Pure Fock state |0⟩_q ⊗ |3⟩_c
        state = tensor(basis(2, 0), basis(max_ph, 3))
        state = state * state.dag()
        dist = number_dist(state, max_ph)
        assert abs(dist.sum() - 1.0) < 1e-10, "number_dist should sum to 1"
        assert abs(dist[3] - 1.0) < 1e-10, "Fock state |3⟩ should have P(3)=1"

    def test_proj_rk_is_projector(self):
        """P² = P for proj_rk."""
        r, k, max_ph = 3, 0, 12
        P = proj_rk(r, k, max_ph)
        diff = (P * P - P).norm()
        assert diff < 1e-10, "proj_rk should be idempotent"

    def test_proj_rk_selects_correct_modes(self):
        """Only photon numbers ≡ k (mod r) should be in the support."""
        r, k, max_ph = 4, 1, 20
        P = proj_rk(r, k, max_ph)
        for n in range(max_ph):
            state = tensor(basis(2, 0), basis(max_ph, n))
            prob = expect(P, state * state.dag())
            if n % r == k:
                assert abs(prob - 1.0) < 1e-10, f"|{n}⟩ should be in support"
            else:
                assert abs(prob) < 1e-10, f"|{n}⟩ should not be in support"


# ---------------------------------------------------------------------------
# Collapse operators
# ---------------------------------------------------------------------------

class TestCOPS:
    def test_c_ops_count(self):
        """C_OPS should return exactly 5 collapse operators."""
        c_ops = C_OPS(0.01, 0.0, 0.005, 0.02, 1000, 10)
        assert len(c_ops) == 5

    def test_c_ops_correct_dims(self):
        max_ph = 8
        c_ops = C_OPS(0.01, 0.0, 0.005, 0.02, 1000, max_ph)
        for op in c_ops:
            assert op.dims == [[2, max_ph], [2, max_ph]]
