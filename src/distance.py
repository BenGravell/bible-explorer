"""Distance functions."""

import gzip
from functools import cache
from joblib import Parallel, delayed

import numpy as np
import streamlit as st


@cache
def glen(x):
    """Length of gzip-compressed data."""
    return len(gzip.compress(x.encode()))


def tcat(x, y):
    """Text concatenation."""
    return " ".join([x, y])


def ncd(lx, ly, lxy):
    """Normalized Compression Distance."""
    return (lxy - min(lx, ly)) / max(lx, ly)


@cache
def gdist(x, y):
    """Distance using gzip compression.

    See https://arxiv.org/abs/2212.09410
    """
    return ncd(glen(x), glen(y), glen(tcat(x, y)))


@st.cache_data(max_entries=10, show_spinner=True)
def compute_distance_matrix(df, labels):
    n = len(df)
    D = np.zeros((n, n))

    def compute_distance(i, j):
        x = df.iloc[i]
        y = df.iloc[j]
        return i, j, gdist(x, y)

    results = Parallel(n_jobs=-1)(delayed(compute_distance)(i, j) for i in range(n) for j in range(i + 1, n))

    for i, j, d in results:
        D[i, j] = d
        D[j, i] = d

    return D
