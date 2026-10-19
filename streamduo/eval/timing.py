"""Timing metrics for streaming commentary (Proact-VL appendix C.1).

Inputs per clip, all times in absolute video seconds:
    gt:   list of (start, end) 1-second segments spoken by the target (role=assistant)
    pred: list of seconds t at which the model spoke (non-silent output in [t, t+1))

Consecutive 1-second segments are merged into utterances (intervals) before matching.

Two TimeDiff variants:
  * timediff()           per-utterance reading of the paper's formula (one GT interval per utterance)
  * timediff_proactvl()  what Proact-VL's released evaluation actually computes (verified clip-by-clip
                         against oaaoaa/proactvl_results my_evaluation/f1): ONE interval per clip,
                         [first GT speaking second, last GT speaking second]; base = first prediction
                         at/after the GT onset - GT onset (span length if none); penalty = alpha *
                         #predicted seconds outside [first-k, last+k]; k=2, alpha=0.2.
F1 likewise is reported both per-clip-averaged ("macro", Proact-VL's headline number) and micro.
"""
from dataclasses import dataclass


def merge_seconds(seconds, max_gap=0):
    """Merge sorted integer seconds into [start, end) intervals; gaps <= max_gap are bridged."""
    out = []
    for s in sorted(set(seconds)):
        if out and s <= out[-1][1] + max_gap:
            out[-1][1] = s + 1
        else:
            out.append([s, s + 1])
    return [tuple(x) for x in out]


def gt_seconds(annotations, role="assistant"):
    secs = []
    for a in annotations:
        if a["role"] == role:
            secs.extend(range(int(a["start"]), int(a["end"])))
    return secs


def timediff_proactvl(gt_secs, pred_secs, k=2, alpha=0.2):
    """Proact-VL's released TimeDiff for one clip. Returns None if the clip has no GT speech."""
    if not gt_secs:
        return None
    first, last = min(gt_secs), max(gt_secs)
    # Released behaviour: the response is the first prediction at/after the GT onset, even when it
    # falls after the GT span; only if there is none does the span length apply.
    after = sorted(t for t in pred_secs if t >= first)
    base = (after[0] - first) if after else (last - first)
    out_of_range = sum(1 for t in pred_secs if not (first - k <= t <= last + k))
    return base + alpha * out_of_range


def timediff(gt_intervals, pred_starts, delta=2.0, alpha=1.0):
    """Mean TimeDiff over GT intervals (lower is better). Returns None if no GT interval."""
    if not gt_intervals:
        return None
    # Associate each prediction with its nearest GT interval (by distance to the interval).
    assoc = {i: [] for i in range(len(gt_intervals))}
    for t in pred_starts:
        dist = [0.0 if a <= t < b else min(abs(t - a), abs(t - b)) for a, b in gt_intervals]
        assoc[min(range(len(dist)), key=dist.__getitem__)].append(t)
    total = 0.0
    for i, (a, b) in enumerate(gt_intervals):
        inside = [t for t in pred_starts if a <= t < b]
        base = min(abs(t - a) for t in inside) if inside else (b - a)
        penalty = sum(1 for t in assoc[i] if not (a - delta <= t < b + delta))
        total += base + alpha * penalty
    return total / len(gt_intervals)


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def timeline_prf(gt_secs, pred_secs, begin, end):
    """Per-second precision/recall over the clip timeline [begin, end)."""
    g = {s for s in gt_secs if begin <= s < end}
    p = {s for s in pred_secs if begin <= s < end}
    return PRF(tp=len(g & p), fp=len(p - g), fn=len(g - p))


def clip_timing(row, pred_secs):
    """All timing metrics for one benchmark row and the model's spoken seconds."""
    gs = gt_seconds(row["annotations"])
    return {
        "timediff": timediff_proactvl(gs, pred_secs),
        "timediff_utt": timediff(merge_seconds(gs), [a for a, _ in merge_seconds(pred_secs)]),
        "prf": timeline_prf(gs, pred_secs, row["video_begin"], row["video_end"]),
    }


def aggregate(clip_results):
    """Headline numbers follow Proact-VL: mean TimeDiff over clips with GT speech, per-clip-averaged
    P/R/F1 (macro); micro F1 and the per-utterance TimeDiff are reported alongside."""
    tds = [c["timediff"] for c in clip_results if c["timediff"] is not None]
    tdu = [c["timediff_utt"] for c in clip_results if c["timediff_utt"] is not None]
    micro = PRF()
    for c in clip_results:
        micro.add(c["prf"])
    n = max(len(clip_results), 1)
    return {
        "TimeDiff": sum(tds) / len(tds) if tds else None,
        "Precision": sum(c["prf"].precision for c in clip_results) / n,
        "Recall": sum(c["prf"].recall for c in clip_results) / n,
        "F1": sum(c["prf"].f1 for c in clip_results) / n,
        "F1_micro": micro.f1,
        "TimeDiff_utt": sum(tdu) / len(tdu) if tdu else None,
        "n_clips": len(clip_results),
    }
