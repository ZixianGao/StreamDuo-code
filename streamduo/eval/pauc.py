"""PAUC (ProactiveVideoQA, arXiv 2507.09313 Eq. 1 + §3.1) as used by Proact-VL.

Per reply turn with timespan (t_start, t_end) and judged responses (tau_p, s_p), s_p in {0,1,2}:
  timestamps are pulled left by omega: tau' = t_start + (1 - omega) * (tau - t_start)
  area = (tau'_1 - t_start) * start + sum_p (tau'_{p+1} - tau'_p) * s_p + (t_end - tau'_P) * s_P
  PAUC = area / ((t_end - t_start) * S),  S = 2
Proact-VL uses start = 0 and reports omega = 0.5. Verified against the released per-example judge
scores (oaaoaa/proactvl_results my_evaluation/pauc): omega=0 and the headline omega=0.5 match the
released summary within 0.001; at omega=1 the released numbers are exactly linear in omega, i.e. their
implementation of the omega shift differs from the paper formula at that end (not used for headlines).

The judge prompt below is Proact-VL's, verbatim (from their released judge requests).
"""

JUDGE_SYSTEM = (
    "You are an evaluator for a gaming commentary system. Your task is to rate the whether the predicted "
    "answer covers the key points of the ground truth answer. Use the following scale to assign a score:\n"
    "- 3: Mostly covered; the predicted answer covers all key information in the ground truth answer, though "
    "it may have minor inaccuracies or rephrases.\n"
    "- 2: Partially covered; the predicted answer has some correct information, but also contains significant "
    "inaccuracies or missing key points.\n"
    "- 1: Incorrect; the predicted answer may be related to the ground truth answer, but most of the "
    "information is missing, or the predicted answer is of very poor quality."
    "The similarity between past and current Predicted Answers must be considered, and penalties will be "
    "applied to both cases.Output the score only, do not add more explanations."
)
S_MAX = 2


def judge_user_message(gold, pred):
    return f"Ground Truth Answer: {gold}\nPredicted Answer: {pred}"


def raw_to_score(raw):
    """Judge outputs 1-3; PAUC uses 0-2."""
    digits = [c for c in str(raw) if c in "123"]
    return int(digits[0]) - 1 if digits else 0


def pauc_turn(t_start, t_end, judged, omega=0.5, start=0.0):
    """judged: list of (tau, score in 0..S_MAX) for responses inside (t_start, t_end)."""
    span = t_end - t_start
    if span <= 0:
        return None
    pts = sorted((t_start + (1 - omega) * (t - t_start), s) for t, s in judged if t_start <= t <= t_end)
    area, cur, last = 0.0, start, t_start
    for t, s in pts:
        area += cur * (t - last)
        cur, last = s, t
    area += cur * (t_end - last)
    return area / (span * S_MAX)
