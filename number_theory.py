"""
number_theory.py
----------------
Number-theoretic utilities for the Chinese Remainder Theorem (CRT) based
photon-number measurement protocol.

The core task is to find two coprime integers (r1, r2) such that:
    • r1 * r2  exceeds the required discrimination range σ ~ 6√nbar
    • The residues k_i = nbar mod r_i  are 'well separated' from residues
      of nearby photon numbers, guaranteeing unambiguous state discrimination.
"""

import numpy as np
from math import gcd


# ---------------------------------------------------------------------------
# Prime utilities
# ---------------------------------------------------------------------------

def coprime(a, b):
    """Return True if a and b are coprime (gcd == 1)."""
    return gcd(int(a), int(b)) == 1


def nth_prime(n):
    """
    Return the n-th prime number (1-indexed, so nth_prime(1) = 2).

    Uses trial division; sufficient for the small primes needed here.
    """
    prime_list = [2]
    num = 3
    while len(prime_list) < n:
        for p in prime_list:
            if num % p == 0:
                break
        else:
            prime_list.append(num)
        num += 2
    return int(prime_list[-1])


def prime_factors(nbar):
    """
    Return successive primes whose product first exceeds nbar.

    Parameters
    ----------
    nbar : int

    Returns
    -------
    factor_list : ndarray
    product     : int
    """
    factor_list = [1]
    prime_no = 1
    while np.prod(factor_list) < nbar:
        factor = nth_prime(prime_no)
        factor_list = np.append(factor_list, factor)
        prime_no += 1
    return factor_list, int(np.prod(factor_list))


# ---------------------------------------------------------------------------
# CRT pair search
# ---------------------------------------------------------------------------

def rk_find_best_prime(nbar, m):
    """
    Find the 'best' pair of coprime integers (r1, r2) for a two-round
    CRT photon-number measurement targeting |nbar⟩.

    The search criteria are:
        1. r1 > r2  (ordered pair)
        2. gcd(r1, r2) = 1
        3. r1 * r2 > σ  where σ = 6√nbar
        4. The minimum spacing between residue lists of the two moduli
           around nbar must be ≥ m  (ensures unambiguous discrimination).

    Among all valid pairs, the one minimising r1² + r2² + k1² + k2²
    is returned (prefers small r's and small residues, which means
    shorter pulse sequences).

    Parameters
    ----------
    nbar : int
        Target photon number.
    m : int
        Minimum discrimination gap (typically 3–4).

    Returns
    -------
    product : int
        r1 * r2.
    factors : ndarray([r1, r2])
    residues : ndarray([k1, k2])   where k_i = nbar mod r_i.

    If no valid pair is found, returns the string 'NA'.
    """
    sigma = 6 * np.sqrt(nbar)
    hook = int(1.5 * np.ceil(np.sqrt(sigma)))
    candidates = np.arange(int(np.floor(hook / 2)), int(2 * hook))

    check_list = []
    for i in candidates:
        for j in candidates:
            if j >= i:
                continue
            if not coprime(i, j):
                continue
            if i * j <= sigma:
                continue

            # Build residue grids around nbar for each modulus
            width_i = int(np.ceil(sigma / i))
            width_j = int(np.ceil(sigma / j))
            list_i = np.arange(nbar - width_i * i, nbar + width_i * i, i)
            list_j = np.arange(nbar - width_j * j, nbar + width_j * j, j)

            # For each element of list_j, find the closest in list_i
            diff_list = []
            for l_val in list_j:
                closest = min(list_i, key=lambda x: abs(x - l_val))
                diff_list.append(abs(closest - l_val))
            diff_list = np.sort(diff_list)

            if diff_list[1] >= m:
                check_list.append([i, j, nbar % i, nbar % j])

    if not check_list:
        return 'NA'

    # Sort by cost: prefer small moduli and small residues
    check_list.sort(key=lambda x: x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2)
    best = check_list[0]
    return (
        int(best[0] * best[1]),
        np.array([best[0], best[1]]),
        np.array([best[2], best[3]]),
    )
