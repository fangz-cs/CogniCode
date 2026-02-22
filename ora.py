"""
Optimal Round Adherence (ORA).

Measures whether the number of clarification rounds is near-optimal,
using a Gaussian-shaped penalty around the optimal round count K.
"""

import math


def calculate_ora(
    n: int,
    K: int,
    sigma: float | None = None,
) -> float:
    """
    Compute Optimal Round Adherence (ORA).

    ORA is highest when the number of clarification rounds n equals the
    optimal count K, and decreases smoothly as |n - K| increases. The
    score is in (0, 1].

    Parameters
    ----------
    n : int
        Actual number of rounds in which the model asks clarification
        questions.
    K : int
        Optimal number of rounds (e.g. |Q| + 1 where Q is the set of
        key questions, with the extra round for stopping and transitioning
        to code generation).
    sigma : float, optional
        Scale of the Gaussian penalty. If None, set so that ORA = 0.5
        when |n - K| = 0.5*K, i.e. sigma = 0.5*K / sqrt(2*ln(2)) ≈ 0.425*K.

    Returns
    -------
    float
        ORA in (0, 1]. 1.0 when n == K; approaches 0 as |n - K| grows.
    """
    if K < 0 or n < 0:
        raise ValueError("n and K must be non-negative")

    if sigma is None:
        # ORA = 0.5 when |n - K| = 0.5*K  =>  exp(-(0.5*K)^2/(2*sigma^2)) = 0.5
        # (0.5*K)^2 / (2*sigma^2) = ln(2)  =>  sigma = 0.5*K / sqrt(2*ln(2))
        if K == 0:
            sigma = 0.3  # arbitrary small scale when there are no key questions
        else:
            sigma = (0.5 * K) / math.sqrt(2.0 * math.log(2.0))  # ≈ 0.425*K

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    exponent = -((n - K) ** 2) / (2.0 * sigma**2)
    return math.exp(exponent)
