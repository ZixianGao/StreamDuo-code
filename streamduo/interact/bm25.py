"""Okapi BM25 with an inverted index (pure Python): only documents containing a query term are scored,
so a query costs O(postings) instead of O(corpus). Used for the research agent's wiki search."""
import heapq
import math
from collections import Counter, defaultdict


class BM25Okapi:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(self.n, 1)
        self.post = defaultdict(list)        # term -> [(doc, tf)]
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                self.post[t].append((i, f))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def _sparse(self, query):
        acc = defaultdict(float)
        for q in query:
            for i, f in self.post.get(q, ()):
                acc[i] += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return acc

    def get_scores(self, query):
        out = [0.0] * self.n
        for i, s in self._sparse(query).items():
            out[i] = s
        return out

    def top_k(self, query, k):
        """[(doc, score)] of the k best documents with a positive score."""
        return heapq.nlargest(k, self._sparse(query).items(), key=lambda x: x[1])
