"""Grilele de experimente (--grid, --grid2) și sumarul lor."""

import numpy as np

from .data import inject, layout, make_series
from .metrics import echo_ratio, exceedance, split_q, width_change


def run(forecasters, period=24, n_periods_ctx=8, horizon=None, kinds=("spike", "burst"),
        amps=(3, 5, 10), ks=(1, 2), n_series=30, offset=5, burst_len=6, configs=None):
    """Întoarce o listă de rânduri (dict) cu raportul de ecou per serie.

    configs: listă (kind, L, offset). Dacă lipsește, se construiește din
    kinds / burst_len / offset, ca înainte (lungimi fixe în pași)."""
    horizon = horizon or period
    C = n_periods_ctx * period
    if configs is None:
        configs = [(kind, 1 if kind == "spike" else burst_len, offset) for kind in kinds]
    rows = []
    for kind, L, off in configs:
        for amp in amps:
            for k in ks:
                start, lo, hi = layout(C, period, k, off, L)
                assert hi <= horizon and start + L <= C, "anomalia/fereastra ar fi tăiată"
                clean, anom = [], []
                for s in range(n_series):
                    y = make_series(C, period, seed=s)
                    clean.append(y)
                    anom.append(inject(y, start, L, kind, amp))
                clean, anom = np.array(clean), np.array(anom)
                # numitorul = abaterea medie injectată efectiv (nu vârful),
                # ca seasonal naive să dea exact 1 și pentru rafale
                inj = (anom[:, start:start + L] - clean[:, start:start + L]).mean(axis=1)
                for name, f in forecasters.items():
                    med_c, q10_c, q90_c = split_q(f(clean, horizon))
                    med_a, q10_a, q90_a = split_q(f(anom, horizon))
                    for i in range(n_series):
                        r_in, r_abs, r_out = echo_ratio(med_a[i], med_c[i], lo, hi, inj[i])
                        row = dict(model=name, period=period, kind=kind, L=L,
                                   len_frac=round(L / period, 3), offset=off,
                                   amp=amp, k=k, series=i, echo=r_in,
                                   echo_abs=r_abs, control=r_out,
                                   echo_q90=np.nan, width_in=np.nan, width_out=np.nan,
                                   width_abs_in=np.nan, width_abs_out=np.nan)
                        if q90_c is not None:   # doar TSFM-urile au interval
                            row["echo_q90"] = echo_ratio(q90_a[i], q90_c[i], lo, hi, inj[i])[0]
                            (row["width_in"], row["width_out"],
                             row["width_abs_in"], row["width_abs_out"]) = width_change(
                                q10_a[i], q90_a[i], q10_c[i], q90_c[i], lo, hi, inj[i])
                        rows.append(row)
    return rows


def summarize(rows):
    """Media ecoului + cât din serii au ecou mare (> 0,5).

    A doua coloană deosebește două mecanisme cu aceeași medie:
      - mediere pe perioade: fiecare serie are ecou ~0,125, deci frac_high = 0
      - copiere (parroting): unele serii au ecou 1, altele 0, deci frac_high > 0
    """
    import pandas as pd
    df = pd.DataFrame(rows)
    df["high"] = df["echo"] > 0.5
    keys = [c for c in ["model", "period", "kind", "L", "k", "amp"] if c in df]
    agg = dict(echo=("echo", "mean"), echo_abs=("echo_abs", "mean"),
               frac_high=("high", "mean"), control=("control", "mean"))
    for c in ("echo_q90", "width_in", "width_out", "width_abs_in", "width_abs_out"):
        if c in df:
            agg[c] = (c, "mean")
    return df.groupby(keys).agg(**agg).round(3)


def add_exceedance(df, n_periods_ctx=8):
    """Adaugă exceed_frac și exceed_max la fiecare rând din grid2.csv.

    Nu rulează niciun model: seriile se refac exact din seed (make_series și
    inject sunt deterministe). Depășirea nu depinde de model, deci o
    calculăm o singură dată per (P, tip, L, offset, amp, k, serie)."""
    import pandas as pd
    keys = ["period", "kind", "L", "offset", "amp", "k", "series"]
    rows = []
    for P, kind, L, off, amp, k, s in df[keys].drop_duplicates().itertuples(index=False):
        C = n_periods_ctx * P
        start, _, _ = layout(C, P, k, off, L)
        clean = make_series(C, P, seed=s)
        anom = inject(clean, start, L, kind, amp)
        ef, em = exceedance(clean, anom, start, L)
        rows.append(dict(period=P, kind=kind, L=L, offset=off, amp=amp, k=k, series=s,
                         exceed_frac=ef, exceed_max=em))
    return df.merge(pd.DataFrame(rows), on=keys, how="left")
