"""Teste de control: persistență vs. copiere de fază, granițe de patch (--persist)."""

import numpy as np

from .data import inject, layout, make_series
from .metrics import split_q


def horizon_profile(forecaster, period, kind, L, offset, k=1, amp=5, n_series=100,
                    n_periods_ctx=8):
    """d(h) pe TOT orizontul, pentru fiecare serie, nu doar media pe fereastră.

    Întoarce dict cu:
      d    (n, h): (mediana_anom − mediana_curată) / abaterea injectată
      dw   (n, h): (lățime_anom − lățime_curată) / abaterea injectată, sau None
      level (n,):  valoarea medie a seriei CURATE sub anomalie
                   (< 0 = anomalia cade pe „vale”, > 0 = pe „vârf”)
      lo, hi:      fereastra de ecou (fază) în orizont"""
    C, h = n_periods_ctx * period, period
    start, lo, hi = layout(C, period, k, offset, L)
    assert hi <= h and start + L <= C, "anomalia/fereastra ar fi tăiată"
    clean = np.array([make_series(C, period, seed=s) for s in range(n_series)])
    anom = np.array([inject(y, start, L, kind, amp) for y in clean])
    inj = (anom[:, start:start + L] - clean[:, start:start + L]).mean(axis=1)
    med_c, q10_c, q90_c = split_q(forecaster(clean, h))
    med_a, q10_a, q90_a = split_q(forecaster(anom, h))
    d = (med_a - med_c) / inj[:, None]
    dw = None
    if q10_c is not None:
        dw = ((q90_a - q10_a) - (q90_c - q10_c)) / inj[:, None]
    level = clean[:, start:start + L].mean(axis=1)
    return dict(d=d, dw=dw, level=level, lo=lo, hi=hi)


def persistence_test(forecasters, periods=(24, 48), amp=5, n_series=100):
    """Rafală de P/2 la k=1, mutată astfel încât între sfârșitul ei și origine
    să rămână gap = 0, P/8, P/4, P/2 pași.

    - Copiere de fază: efectul se mută odată cu fereastra [offset, offset+L)
      și are aceeași mărime la orice gap.
    - Persistență (modelul continuă ultimele valori): efectul stă la începutul
      orizontului și scade cu cât gap-ul crește.
    Întoarce (rânduri per serie, curbe medii d(h))."""
    series_rows, curve_rows = [], []
    for P in periods:
        L = P // 2
        for gap in (0, P // 8, P // 4, P // 2):
            off = P - L - gap
            for name, f in forecasters.items():
                r = horizon_profile(f, P, "burst", L, off, k=1, amp=amp, n_series=n_series)
                d, dw = r["d"], r["dw"]
                early = max(1, P // 8)
                for i in range(n_series):
                    series_rows.append(dict(
                        test="distanta", model=name, period=P, L=L, offset=off, gap=gap,
                        series=i, level=r["level"][i],
                        echo_phase=d[i, off:off + L].mean(),   # fereastra de fază
                        echo_start=d[i, :L].mean(),            # primii L pași
                        echo_early=d[i, :early].mean(),        # primii P/8 pași
                        dw_phase=np.nan if dw is None else dw[i, off:off + L].mean()))
                se = d.std(axis=0, ddof=1) / np.sqrt(n_series)
                for hh in range(P):
                    curve_rows.append(dict(
                        model=name, period=P, L=L, offset=off, gap=gap, h=hh,
                        d_mean=d[:, hh].mean(), d_se=se[hh],
                        dw_mean=np.nan if dw is None else dw[:, hh].mean()))
    return series_rows, curve_rows


def patch_test(forecasters, periods=(24, 48), amp=5, n_series=100, shifts=range(-3, 4)):
    """Rafală de P/2 la k=1, cu offset-ul de bază (P/5) deplasat cu −3..+3 pași.

    Dacă ecoul ar ține de granițele dintre patch-uri (16 pași la Chronos),
    ar trebui să sară brusc la anumite offset-uri. Atenție: granițele sunt
    aceleași pentru toate seriile, deci nu pot explica diferențele dintre
    serii; acelea vin din fază (vezi coloana level)."""
    rows = []
    for P in periods:
        L, base = P // 2, round(0.2 * P)
        for sh in shifts:
            off = base + sh
            C = 8 * P
            start = C - P + off
            for name, f in forecasters.items():
                r = horizon_profile(f, P, "burst", L, off, k=1, amp=amp, n_series=n_series)
                for i in range(n_series):
                    rows.append(dict(
                        test="patch", model=name, period=P, L=L, offset=off, shift=sh,
                        start_mod16=start % 16, series=i, level=r["level"][i],
                        echo_phase=r["d"][i, off:off + L].mean()))
    return rows
