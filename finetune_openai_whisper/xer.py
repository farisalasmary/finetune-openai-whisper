"""
Word Error Rate (WER) and Character Error Rate (CER) computation.

Both metrics are based on Levenshtein edit distance and follow the
standard ASR evaluation convention: lower is better, 0 = perfect match.

The edit-distance implementation is pure Python (no external C extension),
so this module has no extra dependencies beyond NumPy.
"""

import numpy as np
from typing import List, Tuple, Union


# ── Core edit-distance algorithm ──────────────────────────────────────────────

def _edit_distance_with_ops(
    ref: List,
    hyp: List,
) -> Tuple[int, List[Tuple[str, Union[int, None], Union[int, None]]]]:
    """
    Compute the Levenshtein edit distance and the alignment operation sequence.

    Uses standard dynamic programming (Wagner-Fischer algorithm) followed by
    backtracking to reconstruct the optimal edit path.

    Args:
        ref: Reference sequence (list of tokens or characters).
        hyp: Hypothesis sequence (list of tokens or characters).

    Returns:
        Tuple of (edit_distance, operations), where operations is a list of
        (op, ref_idx, hyp_idx) triples:
          - ``'equal'``      — tokens matched; both indices are valid.
          - ``'substitute'`` — ref[ref_idx] replaced by hyp[hyp_idx].
          - ``'delete'``     — ref[ref_idx] deleted; hyp_idx is None.
          - ``'insert'``     — hyp[hyp_idx] inserted; ref_idx is None.
    """
    m, n = len(ref), len(hyp)

    # DP table: dp[i][j] = edit distance between ref[:i] and hyp[:j].
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack to recover the edit operations.
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            ops.append(('equal', i - 1, j - 1))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(('substitute', i - 1, j - 1))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(('delete', i - 1, None))
            i -= 1
        else:
            ops.append(('insert', None, j - 1))
            j -= 1

    ops.reverse()
    return dp[m][n], ops


def _count_ops(ops: List[Tuple]) -> dict:
    """
    Summarise an operation list into insertion/deletion/substitution counts.

    Returns:
        Dict with keys ``'ins'``, ``'del'``, ``'sub'``, and ``'total'``.
    """
    counts = {'ins': 0, 'del': 0, 'sub': 0}
    for op, _, _ in ops:
        if   op == 'substitute': counts['sub'] += 1
        elif op == 'insert':     counts['ins'] += 1
        elif op == 'delete':     counts['del'] += 1
    counts['total'] = counts['ins'] + counts['del'] + counts['sub']
    return counts


def _edit_distance(ref, hyp) -> dict:
    """
    Return edit-distance statistics between two sequences.

    Accepts strings (compared character-by-character) or lists of tokens.

    Returns:
        Dict with keys ``'ins'``, ``'del'``, ``'sub'``, ``'total'``.
    """
    if isinstance(ref, str):
        ref = list(ref)
    if isinstance(hyp, str):
        hyp = list(hyp)
    _, ops = _edit_distance_with_ops(ref, hyp)
    return _count_ops(ops)


# ── Public API ────────────────────────────────────────────────────────────────

def cer(ref: str, hyp: str) -> dict:
    """
    Compute Character Error Rate between a reference and hypothesis string.

    Spaces are removed before comparison so CER measures character-level
    accuracy independently of word segmentation.

    Args:
        ref: Ground-truth transcription.
        hyp: Model-predicted transcription.

    Returns:
        Dict with keys:
          - ``'insertions'``, ``'deletions'``, ``'substitutions'`` — raw counts.
          - ``'distance'``   — total edit distance (ins + del + sub).
          - ``'ref_length'`` — number of reference characters (denominator).
          - ``'Error Rate'`` — CER as a percentage (distance / ref_length * 100).
    """
    ref_chars = ref.replace(' ', '').strip()
    hyp_chars = hyp.replace(' ', '').strip()
    info      = _edit_distance(ref_chars, hyp_chars)

    return {
        'insertions':    info['ins'],
        'deletions':     info['del'],
        'substitutions': info['sub'],
        'distance':      info['total'],
        'ref_length':    float(len(ref_chars)),
        'Error Rate':    (info['total'] / len(ref_chars)) * 100 if ref_chars else 0.0,
    }


def wer(ref: str, hyp: str) -> dict:
    """
    Compute Word Error Rate between a reference and hypothesis string.

    Each string is tokenised by whitespace. Words are mapped to unique
    single characters before computing edit distance so the same
    character-level DP can be reused without modification.

    Args:
        ref: Ground-truth transcription.
        hyp: Model-predicted transcription.

    Returns:
        Dict with keys:
          - ``'insertions'``, ``'deletions'``, ``'substitutions'`` — raw counts.
          - ``'distance'``   — total edit distance (ins + del + sub).
          - ``'ref_length'`` — number of reference words (denominator).
          - ``'Error Rate'`` — WER as a percentage (distance / ref_length * 100).
    """
    ref_words = ref.split()
    hyp_words = hyp.split()

    # Map each unique word to a single character so the edit-distance DP
    # operates on characters (required by the string-based interface).
    vocab     = set(ref_words + hyp_words)
    word2char = {w: chr(i) for i, w in enumerate(vocab)}

    ref_enc = [word2char[w] for w in ref_words]
    hyp_enc = [word2char[w] for w in hyp_words]

    info = _edit_distance(ref_enc, hyp_enc)

    return {
        'insertions':    info['ins'],
        'deletions':     info['del'],
        'substitutions': info['sub'],
        'distance':      info['total'],
        'ref_length':    float(len(ref_words)),
        'Error Rate':    (info['total'] / len(ref_words)) * 100 if ref_words else 0.0,
    }
