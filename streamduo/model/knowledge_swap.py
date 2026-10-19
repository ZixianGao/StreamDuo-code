"""Knowledge-Swap augmentation.

With probability p per training sample, game-specific entity names are renamed CONSISTENTLY in
every text the model sees or predicts -- dossier, target commentary, partner lines, persona,
history -- to pseudo-words. To say the right name the model must then map what it sees to the
dossier entry (via its visual_cue/facts) and copy the name, instead of recalling it from weights.

Which names are swapped: entities and people the dossier lists (terminology only with
swap_terms=True, since much jargon is shared across games), plus gold-ontology entities
(non-generic, typed) that occur in the clip's texts. The game title itself is never swapped.
Pseudo-names are built from syllables, seeded by (sample seed, original name), with the
original's word count and capitalisation pattern.
"""
import json
import random
import re

from streamduo.wiki.ontology import EntityMatcher

SYLLABLES = ["ka", "ro", "vel", "tha", "mir", "zen", "dor", "lua", "quen", "sar", "bri", "oth",
             "vy", "ral", "nex", "ith", "gor", "fae", "mon", "tir", "esh", "dra", "kel", "uma"]
SWAP_TYPES = {"CHARACTER", "ITEM", "ABILITY", "LOCATION", "PLAYER", "TEAM", "EVENT"}


def pseudo_word(rng, n_syl=None):
    n = n_syl or rng.choice([2, 2, 3])
    return "".join(rng.choice(SYLLABLES) for _ in range(n)).capitalize()


def pseudo_name(name, rng):
    words = name.split()
    out = []
    for w in words:
        if w.lower() in {"of", "the", "and", "de", "von", "la", "le"}:
            out.append(w)
        elif re.fullmatch(r"\W+|\d+", w):
            out.append(w)
        else:
            p = pseudo_word(rng)
            out.append(p if w[:1].isupper() else p.lower())
    return " ".join(out)


class KnowledgeSwap:
    def __init__(self, p=0.5, min_len=4, swap_terms=False):
        self.p = p
        self.min_len = min_len
        self.swap_terms = swap_terms
        self._matchers = {}

    def _matcher(self, game):
        if game not in self._matchers:
            try:
                self._matchers[game] = EntityMatcher.for_game(game)
            except FileNotFoundError:
                self._matchers[game] = None
        return self._matchers[game]

    def _dossier_names(self, dossier_json):
        names = [e.get("name", "") for e in dossier_json.get("entities", []) if isinstance(e, dict)]
        names += [p.get("name", "") for p in dossier_json.get("people", []) if isinstance(p, dict)]
        if self.swap_terms:
            names += [t.get("term", "") for t in dossier_json.get("terminology", []) if isinstance(t, dict)]
        return [re.sub(r"\s*\([^)]*\)$", "", n).strip() for n in names]

    def __call__(self, row, dossier, rng, dossier_json=None, game_title=""):
        """Returns (row', dossier') with consistent renames; unchanged with prob 1-p."""
        if rng.random() >= self.p:
            return row, dossier
        texts = [dossier or "", row["active_speaker"].get("persona", "") or "", row.get("history", "") or ""]
        texts += [a.get("text", "") or a.get("query", "") for a in row["annotations"]]
        blob = "\n".join(t if isinstance(t, str) else json.dumps(t) for t in texts)

        names = set(self._dossier_names(dossier_json or {}))
        m = self._matcher(row["metadata"]["dataset_name"])
        if m is not None:
            names |= {re.sub(r"\s*\([^)]*\)$", "", e["name"]) for e in m.find(blob) if e["type"] in SWAP_TYPES}
        title_l = game_title.lower()
        names = sorted({n for n in names if len(n) >= self.min_len and n.lower() not in title_l
                        and re.search(rf"(?<!\w){re.escape(n)}(?!\w)", blob, re.I)}, key=len, reverse=True)
        if not names:
            return row, dossier
        mapping = {n: pseudo_name(n, random.Random(f"{rng.random()}|{n}")) for n in names}
        pat = re.compile("|".join(rf"(?<!\w){re.escape(n)}(?!\w)" for n in names), re.I)
        lower_map = {k.lower(): v for k, v in mapping.items()}

        def sub(text):
            if not isinstance(text, str) or not text:
                return text
            return pat.sub(lambda mt: lower_map[mt.group(0).lower()], text)

        row = json.loads(json.dumps(row))
        row["active_speaker"]["persona"] = sub(row["active_speaker"].get("persona", ""))
        if row.get("history"):
            row["history"] = sub(row["history"])
        for a in row["annotations"]:
            for k in ("text", "query"):
                if k in a:
                    a[k] = sub(a[k])
        row["_swap"] = mapping
        return row, sub(dossier) if dossier else dossier
