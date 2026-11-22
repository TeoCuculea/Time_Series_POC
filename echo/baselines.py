"""Modele de referință fără GPU: seasonal naive / mean / median, context parroting."""

import numpy as np


def seasonal_naive(ctx, h, period):
    """Repetă ultima perioadă. Ecou așteptat: 1 la k=1, 0 la k=2."""
    last = ctx[-period:]
    reps = int(np.ceil(h / period))
    return np.tile(last, reps)[:h]


def seasonal_mean(ctx, h, period):
    """Media pe faze a tuturor perioadelor din context.
    Ecou așteptat ≈ 1 / (numărul de perioade din context)."""
    n_per = len(ctx) // period
    tail = ctx[-n_per * period:].reshape(n_per, period)
    prof = tail.mean(axis=0)
    reps = int(np.ceil(h / period))
    return np.tile(prof, reps)[:h]


def seasonal_median(ctx, h, period):
    """Varianta robustă: mediana pe faze. Ecou așteptat ≈ 0."""
    n_per = len(ctx) // period
    tail = ctx[-n_per * period:].reshape(n_per, period)
    prof = np.median(tail, axis=0)
    reps = int(np.ceil(h / period))
    return np.tile(prof, reps)[:h]


def context_parrot(ctx, h, D=10):
    """Context parroting (Zhang & Gilpin, Algoritmul 1), fără parametri antrenabili.

    1. Interogarea = ultimele D puncte din context.
    2. Caută în context fereastra de lungime D cea mai apropiată (distanță
       euclidiană) de interogare, fără fereastra-interogare însăși.
    3. Copiază ce a urmat după acea fereastră, repetând până la h puncte.

    De ce contează pentru tine: dacă potrivirea cade cu o perioadă în urmă,
    copia include anomalia de acum o perioadă (ecou = 1); dacă cade mai
    devreme, n-o include (ecou = 0). Ecoul pe serie e deci „totul sau nimic”.
    """
    L = len(ctx)
    query = ctx[L - D:]
    best_end, best_dist = None, np.inf
    for end in range(D, L):              # fereastra ctx[end-D:end] se termină la `end`
        dist = np.sum((ctx[end - D:end] - query) ** 2)
        if dist < best_dist:
            best_dist, best_end = dist, end
    cont = ctx[best_end:]                # tot ce a urmat după potrivire
    reps = int(np.ceil(h / len(cont)))
    return np.tile(cont, reps)[:h]


def batchify(fn, period):
    """Transformă un baseline pe o serie într-unul pe un lot de serii."""
    return lambda batch, h: np.array([fn(c, h, period) for c in batch])
