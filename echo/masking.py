"""H3 (mascarea anomaliei) și costul unei alarme false (--h3, --maskcost)."""

import numpy as np

from .data import inject, layout, make_series, proportional_configs, repair
from .metrics import echo_ratio, split_q, width_change


H3_VARIANTS = ("anomalie", "masca", "interp", "sezonier", "masca_centru50")


def h3_test(forecasters, periods=(24, 48), amps=(3, 5, 10), ks=(1, 2), n_series=100,
            frac=0.5, kind="burst"):
    """Ecoul după fiecare tip de reparație, pe rafala de P/2 (offset P/5).

    Pentru fiecare serie compară prognoza seriei reparate cu prognoza
    geamănului curat NEATINS (fără NaN), normalizat la anomalia injectată."""
    rows = []
    for P in periods:
        _, L, off = proportional_configs(P, [(kind, frac)])[0]
        C = 8 * P
        for amp in amps:
            for k in ks:
                start, lo, hi = layout(C, P, k, off, L)
                clean = np.array([make_series(C, P, seed=s) for s in range(n_series)])
                anom = np.array([inject(y, start, L, kind, amp) for y in clean])
                inj = (anom[:, start:start + L] - clean[:, start:start + L]).mean(axis=1)
                for name, f in forecasters.items():
                    med_c, q10_c, q90_c = split_q(f(clean, P))
                    for how in H3_VARIANTS:
                        rep = np.array([repair(y, start, L, how, P) for y in anom])
                        med_a, q10_a, q90_a = split_q(f(rep, P))
                        for i in range(n_series):
                            e, _, ctrl = echo_ratio(med_a[i], med_c[i], lo, hi, inj[i])
                            w = width_change(q10_a[i], q90_a[i], q10_c[i], q90_c[i],
                                             lo, hi, inj[i])
                            rows.append(dict(model=name, period=P, L=L, offset=off, amp=amp,
                                             k=k, variant=how, series=i, echo=e,
                                             control=ctrl, width_abs_in=w[2]))
    return rows


def h3_summary(df, n_boot=2000, seed=0):
    """Ecoul mediu per variantă + reducerea față de 'anomalie', cu IC bootstrap.

    reducere = 1 − ecou_reparat / ecou_anomalie (pe medii). Bootstrap-ul
    retrage aceleași serii pentru ambele variante (date pereche)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    out = []
    for (m, P, amp, k), g in df.groupby(["model", "period", "amp", "k"]):
        base = g[g["variant"] == "anomalie"].sort_values("series")["echo"].values
        for how in H3_VARIANTS:
            v = g[g["variant"] == how].sort_values("series")["echo"].values
            red = 1 - v.mean() / base.mean()
            idx = rng.integers(0, len(v), size=(n_boot, len(v)))
            reds = 1 - v[idx].mean(axis=1) / base[idx].mean(axis=1)
            out.append(dict(model=m, period=P, amp=amp, k=k, variant=how,
                            echo=v.mean(), reducere=red,
                            red_lo=np.quantile(reds, 0.025), red_hi=np.quantile(reds, 0.975)))
    return pd.DataFrame(out)


def mask_cost_test(forecasters, periods=(24, 48), ks=(1, 2), n_series=100, frac=0.5,
                   fills=("masca", "interp", "sezonier")):
    """Costul unei alarme false: „reparăm” un segment NORMAL de P/2.

    Pe geamănul curat, la aceeași poziție ca la H3 (offset P/5), comparăm
    prognoza seriei întregi cu prognoza după fiecare umplere a golului
    (vezi repair(): masca = NaN, interp = linie dreaptă, sezonier = y[t−P]),
    față de viitorul real al seriei:
      mae_full / mae_<umplere>     - eroarea medie absolută pe tot orizontul
      width_full / width_<umplere> - lățimea medie a intervalului q0.1–q0.9
    Viitorul real vine din aceeași serie generată mai lung (același seed
    dă aceleași prime C valori, verificat)."""
    rows = []
    for P in periods:
        _, L, off = proportional_configs(P, [("burst", frac)])[0]
        C = 8 * P
        full = np.array([make_series(C + P, P, seed=s) for s in range(n_series)])
        ctx, truth = full[:, :C], full[:, C:]
        for k in ks:
            start, lo, hi = layout(C, P, k, off, L)
            filled = {how: np.array([repair(y, start, L, how, P) for y in ctx]) for how in fills}
            for name, f in forecasters.items():
                med_f, q10_f, q90_f = split_q(f(ctx, P))
                out = {how: split_q(f(x, P)) for how, x in filled.items()}
                for i in range(n_series):
                    row = dict(model=name, period=P, k=k, series=i,
                               mae_full=np.abs(med_f[i] - truth[i]).mean(),
                               width_full=(q90_f[i] - q10_f[i]).mean())
                    for how, (med, q10, q90) in out.items():
                        row[f"mae_{how}"] = np.abs(med[i] - truth[i]).mean()
                        row[f"width_{how}"] = (q90[i] - q10[i]).mean()
                    rows.append(row)
    return rows
