import math

import numpy as np
import pytest

from trtbench.accuracy import cosine_vs_ref, perplexity, top1_pct


def test_top1():
    logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    assert top1_pct(logits, np.array([1, 0, 0, 0])) == pytest.approx(75.0)


def test_cosine_identical_and_orthogonal():
    a = np.random.RandomState(0).randn(4, 8, 3)
    assert cosine_vs_ref(a, a) == pytest.approx(1.0)
    b = np.zeros((1, 2)); b[0, 0] = 1
    c = np.zeros((1, 2)); c[0, 1] = 1
    assert cosine_vs_ref(b, c) == pytest.approx(0.0)


def test_uniform_logits_give_vocab_size_perplexity():
    # A model that knows nothing has perplexity equal to the vocabulary size.
    vocab = 50
    logits = np.zeros((3, 16, vocab), dtype=np.float32)
    ids = np.random.RandomState(1).randint(0, vocab, (3, 16))
    assert perplexity(logits, ids, chunk=2) == pytest.approx(vocab, rel=1e-9)


def test_perplexity_matches_direct_computation():
    rng = np.random.RandomState(2)
    logits = rng.randn(2, 5, 7).astype(np.float32)
    ids = rng.randint(0, 7, (2, 5))
    nll = []
    for b in range(2):
        for t in range(4):
            row = logits[b, t].astype(np.float64)
            nll.append(math.log(np.exp(row).sum()) - row[ids[b, t + 1]])
    assert perplexity(logits, ids, chunk=1) == pytest.approx(math.exp(np.mean(nll)))
