"""Metrici: ecou, lățimea intervalului, depășirea intervalului, bootstrap."""

import numpy as np


def echo_ratio(fc_anom, fc_clean, echo_lo, echo_hi, inj_mean):
    """Raportul de ecou pe fereastra așteptată + același calcul în afara ei.

    Întoarce trei numere:
      in_win  - media CU SEMN în fereastră (pozitiv = modelul copiază anomalia)
      in_abs  - media în VALOARE ABSOLUTĂ în fereastră. Dacă in_abs e mare dar
                in_win ~0, modelul a fost afectat, dar abaterile în sus și în
                jos s-au anulat în medie.
      out_win - media în valoare absolută ÎN AFARA ferestrei (control), ~0 ideal.
                Dacă e mare, anomalia a stricat toată prognoza, nu doar faza ei
                („contaminare generală”, nu „ecou”).
    """
    d = (fc_anom - fc_clean) / inj_mean
    in_win = d[echo_lo:echo_hi].mean()
    in_abs = np.abs(d[echo_lo:echo_hi]).mean()
    mask = np.ones_like(d, dtype=bool)
    mask[echo_lo:echo_hi] = False
    # dacă fereastra acoperă tot orizontul (anomalie lungă cât perioada),
    # nu rămâne nimic pentru control -> NaN, nu o medie pe zero puncte
    out_win = np.abs(d[mask]).mean() if mask.any() else np.nan
    return in_win, in_abs, out_win


def width_change(lo_a, hi_a, lo_c, hi_c, echo_lo, echo_hi, inj_mean):
    """Cât de lat e intervalul q0.1–q0.9 cu anomalie față de geamănul curat.

    Întoarce 4 numere, fiecare în fereastra de ecou (in) și în afara ei (out):
      ratio_in / ratio_out: lățime_cu_anomalie / lățime_curată. 1 = neschimbat,
          1.3 = interval cu 30% mai lat. Depinde de cât de îngust era geamănul
          curat: pe serii cu zgomot mic, raportul iese mare ușor.
      abs_in / abs_out: (lățime_cu_anomalie − lățime_curată) / abaterea injectată.
          Pe aceeași scară ca ecoul: 1 = intervalul s-a lărgit cu cât anomalia."""
    w_a, w_c = hi_a - lo_a, hi_c - lo_c
    mask = np.zeros_like(w_a, dtype=bool)
    mask[echo_lo:echo_hi] = True
    has_out = (~mask).any()
    ratio_in = w_a[mask].mean() / w_c[mask].mean()
    ratio_out = w_a[~mask].mean() / w_c[~mask].mean() if has_out else np.nan
    abs_in = (w_a - w_c)[mask].mean() / inj_mean
    abs_out = (w_a - w_c)[~mask].mean() / inj_mean if has_out else np.nan
    return ratio_in, ratio_out, abs_in, abs_out


def split_q(fc):
    """Desparte ieșirea unui model în (mediană, q0.1, q0.9).

    Baseline-urile dau o singură valoare per pas (array 2D), deci n-au
    interval: atunci q0.1 și q0.9 sunt None. TSFM-urile dau 3D (..., 3)."""
    fc = np.asarray(fc)
    if fc.ndim == 3:
        return fc[..., 1], fc[..., 0], fc[..., 2]
    return fc, None, None


def bootstrap_ci(x, n_boot=5000, level=0.95, seed=0):
    """Interval de încredere percentilă pentru media lui x.

    Retrage cu înlocuire n valori din x de n_boot ori, calculează media de
    fiecare dată și ia cuantilele 2.5% / 97.5% ale acestor medii."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return np.quantile(means, a), np.quantile(means, 1 - a)


def exceedance(clean_ctx, anom_ctx, start, L):
    """Cât iese anomalia din intervalul [min, max] al contextului CURAT.

    Întoarce (exceed_frac, exceed_max):
      exceed_frac - fracțiunea din cele L puncte ale anomaliei aflate în afara
                    intervalului (0 = toată anomalia arată „plauzibil”)
      exceed_max  - cea mai mare depășire, în deviații standard ale seriei
                    (0 dacă nu iese deloc)"""
    lo, hi, sd = clean_ctx.min(), clean_ctx.max(), clean_ctx.std()
    seg = anom_ctx[start:start + L]
    out = np.maximum(np.maximum(seg - hi, 0), np.maximum(lo - seg, 0))
    return (out > 0).mean(), out.max() / sd
