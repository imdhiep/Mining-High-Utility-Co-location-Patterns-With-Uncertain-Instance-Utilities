"""Metrics of Section VI-A (defined once in the paper's Experimental Settings)."""


def precision(reported, true):
    """share of the reported patterns that are true; None for an empty answer (shown as "-")"""
    return len(reported & true) / len(reported) if reported else None


def recall(reported, true):
    return len(reported & true) / len(true) if true else None


def top_k(score_of, k):
    """the patterns whose score is at least the k-th largest (ties included); all when fewer than k"""
    if len(score_of) <= k:
        return set(score_of)
    kth = sorted(score_of.values(), reverse=True)[k - 1]
    return {p for p, s in score_of.items() if s >= kth}


def coverage(lower_totals, true_totals):
    """share of the reported patterns whose true total reaches the lower bound of the method"""
    pairs = [(lb, t) for lb, t in zip(lower_totals, true_totals)]
    return sum(t >= lb for lb, t in pairs) / len(pairs) if pairs else None
