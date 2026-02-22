"""
Turn-discounted Key Question Rate (TKQR).

Evaluates how early key clarification questions are covered in a dialogue,
using a normalized discounted cumulative gain over a hit sequence.
"""

import math


def calculate_tkqr(h_sequence: list[int], total_key_questions: int) -> float:
    """
    Compute Turn-discounted Key Question Rate (TKQR).

    TKQR adapts normalized DCG to reward asking key clarification questions
    early. The score is in [0, 1]; it increases when key questions are
    covered earlier and decreases when they are delayed or turns are spent
    on non-key questions.

    Parameters
    ----------
    h_sequence : list[int]
        Hit sequence of length n (number of dialogue turns before the model
        stops asking). Each element is 0 or 1: 1 if the model asked a
        previously uncovered key question at that turn, 0 otherwise.
    total_key_questions : int
        Number of annotated key questions for the task (K).

    Returns
    -------
    float
        TKQR in [0.0, 1.0]. 1.0 iff key questions are covered as early as
        possible; 0.0 if no key question is ever hit (or K == 0).
    """
    n = len(h_sequence)
    if n == 0 or total_key_questions <= 0:
        return 0.0

    # DCG: discounted gain favoring early hits (turn i uses log2(i+1), i 1-based)
    dcg = 0.0
    for i, hit in enumerate(h_sequence):
        dcg += hit / math.log2(i + 2)  # turn index 1-based: i+2 = (i+1)+1

    # IDCG: ideal case — hits in first min(n, K) turns
    ideal_len = min(n, total_key_questions)
    idcg = 0.0
    for i in range(ideal_len):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg
