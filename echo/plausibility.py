"""Ipoteza de plauzibilitate: anomalia în / peste intervalul seriei (--plaus)."""

import numpy as np

from .data import inject, layout, make_series, proportional_configs
from .metrics import echo_ratio, exceedance, split_q, width_change


def position_shift(period, seed, center, j, n_pos=8):
    """Deplasarea care pune centrul anomaliei în punctul j (din n_pos) al ciclului.

    Punctul 0 = valea formei fără zgomot a seriei `seed`; punctele următoare
    sunt la câte P/n_pos pași după vale. Întoarce (shift, level, eticheta):
    level = valoarea formei în acel punct, scalată 0 (vale) .. 1 (vârf)."""
    shape = make_series(period, period, noise=0.0, seed=seed)   # o perioadă, fără zgomot
    v0 = int(np.argmin(shape))
    p = (v0 + j * period // n_pos) % period
    shift = (p - center) % period
    level = (shape[p] - shape.min()) / (shape.max() - shape.min())
    label = "vale" if level < 1 / 3 else ("vârf" if level > 2 / 3 else "pantă")
    return shift, level, label


def amp_to_reach(clean, start, L, target, direction=+1):
    """Amplitudinea (în σ) la care rafala atinge EXACT nivelul `target`.

    Rafala adaugă amp·σ·w[t] (w = clopot Hann). Punctul t atinge ținta când
    amp = (target − clean[t]) / (σ·w[t]); primul punct care o atinge dă
    minimul. direction=−1: aceeași logică pentru o rafală în jos (drop)."""
    w = np.hanning(L + 2)[1:-1]
    sd = clean.std()
    seg = clean[start:start + L]
    return np.min(direction * (target - seg) / (sd * w))


SWEEP_MULTS = (0.25, 0.5, 0.75, 0.95, 1.25, 1.5, 2.0, 3.0)


def plausibility_test(forecasters, periods=(24, 48), n_pos=8, n_series=100,
                      mults=SWEEP_MULTS, out_margin=2.0, drop_fixed=5.0, frac=0.5):
    """Pasul B: aceeași rafală P/2 (offset P/5, k=1), mutată pe 8 puncte ale ciclului.

    La fiecare punct și serie, variante (amplitudini calculate pe serie):
      sweep × m   : m · a_in, unde a_in = amplitudinea care atinge maximul contextului
      in_range    : a_in (atinge maximul, nu-l depășește)
      out_range   : depășește maximul cu out_margin·σ
      drop_in     : rafală în jos care atinge minimul contextului
      drop_out    : rafală în jos sub minim cu out_margin·σ
      drop_fixed  : rafală în jos de drop_fixed·σ („pană de curent”)
    Toate variantele unui punct trec printr-un singur apel de model (lot mare)."""
    rows = []
    for P in periods:
        _, L, off = proportional_configs(P, [("burst", frac)])[0]
        C = 8 * P
        start, lo, hi = layout(C, P, 1, off, L)
        center = start + L // 2
        for j in range(n_pos):
            info, cleans = [], []
            for s in range(n_series):
                sh, level, label = position_shift(P, s, center, j, n_pos)
                y = make_series(C, P, seed=s, shift=sh)
                sd = y.std()
                info.append(dict(series=s, level=level, label=label, sd=sd,
                                 a_in=amp_to_reach(y, start, L, y.max(), +1),
                                 a_out=amp_to_reach(y, start, L, y.max() + out_margin * sd, +1),
                                 d_in=amp_to_reach(y, start, L, y.min(), -1),
                                 d_out=amp_to_reach(y, start, L, y.min() - out_margin * sd, -1)))
                cleans.append(y)
            cleans = np.array(cleans)
            # lista variantelor: (nume, multiplu, direcție, amplitudine per serie)
            variants = [("sweep", m, +1, [m * x["a_in"] for x in info]) for m in mults]
            # 0.999: fără el, rotunjirile scot vârful anomaliei cu ~1e-16 peste maxim
            variants += [("in_range", 1.0, +1, [0.999 * x["a_in"] for x in info]),
                         ("out_range", np.nan, +1, [x["a_out"] for x in info]),
                         ("drop_in", 1.0, -1, [0.999 * x["d_in"] for x in info]),
                         ("drop_out", np.nan, -1, [x["d_out"] for x in info]),
                         ("drop_fixed", np.nan, -1, [drop_fixed] * n_series)]
            batches = [cleans]
            for _, _, sign, amps in variants:
                batches.append(np.array([inject(y, start, L, "burst", sign * a)
                                         for y, a in zip(cleans, amps)]))
            big = np.concatenate(batches)
            for name, f in forecasters.items():
                med, q10, q90 = split_q(f(big, P))
                for v, (vname, mult, sign, amps) in enumerate(variants):
                    b = (v + 1) * n_series            # blocul variantei în lotul mare
                    for i in range(n_series):
                        anom = batches[v + 1][i]
                        inj = (anom[start:start + L] - cleans[i, start:start + L]).mean()
                        with np.errstate(divide="ignore", invalid="ignore"):
                            e, e_abs, ctrl = echo_ratio(med[b + i], med[i], lo, hi, inj)
                            tail_a, tail_c = (q90, q90) if sign > 0 else (q10, q10)
                            e_tail = echo_ratio(tail_a[b + i], tail_c[i], lo, hi, inj)[0]
                            w_abs = width_change(q10[b + i], q90[b + i], q10[i], q90[i],
                                                 lo, hi, inj)[2]
                        ef, em = exceedance(cleans[i], anom, start, L)
                        x = info[i]
                        rows.append(dict(model=name, period=P, pos=j, level=x["level"],
                                         label=x["label"], variant=vname, mult=mult,
                                         direction=sign, amp=sign * amps[i],
                                         a_in=x["a_in"], d_in=x["d_in"], series=i,
                                         echo=e, echo_abs=e_abs, echo_tail=e_tail,
                                         width_abs_in=w_abs, control=ctrl,
                                         exceed_frac=ef, exceed_max=em,
                                         shift_sigma=e * abs(inj) / x["sd"]))
    return rows


def plausibility_summary(df, min_amp=0.5, n_boot=2000, seed=0):
    """Contrastele din regula de decizie, pe (model, P, eticheta poziției).

    Exclude seriile la care amplitudinea „în interval” e < min_amp σ
    (a_in pentru rafala în sus, d_in pentru drop). Întoarce (contraste, excluse):
      in_out    : echo(in_range) − echo(out_range), plus aceeași diferență pe shift_sigma
      drop      : la fel pentru drop_in − drop_out
      kink      : [echo(0.95) − echo(1.25)] − [echo(0.5) − echo(0.75)]  (cădere la prag)
    Toate pe date pereche (aceeași serie), IC bootstrap 95%."""
    import pandas as pd
    rng = np.random.default_rng(seed)

    def ci(x):
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        m = x[idx].mean(axis=1)
        return x.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)

    def wide(g, col):
        return g.pivot_table(index="series", columns="key", values=col)

    out = []
    for (m, P, lab), g in df.groupby(["model", "period", "label"]):
        g = g.assign(key=np.where(g["variant"] == "sweep", "x" + g["mult"].astype(str), g["variant"]))
        up_ok = g[g["a_in"] >= min_amp]
        dn_ok = g[g["d_in"] >= min_amp]
        contrasts = []
        if len(up_ok):
            e, s = wide(up_ok, "echo"), wide(up_ok, "shift_sigma")
            contrasts += [("in_out_echo", (e["in_range"] - e["out_range"]).dropna().values),
                          ("in_out_shift", (s["in_range"] - s["out_range"]).dropna().values),
                          ("kink", ((e["x0.95"] - e["x1.25"]) - (e["x0.5"] - e["x0.75"])).dropna().values)]
        if len(dn_ok):
            e, s = wide(dn_ok, "echo"), wide(dn_ok, "shift_sigma")
            contrasts += [("drop_in_out_echo", (e["drop_in"] - e["drop_out"]).dropna().values),
                          ("drop_in_out_shift", (s["drop_in"] - s["drop_out"]).dropna().values)]
        for cname, x in contrasts:
            if len(x) >= 5:
                mean, lo, hi = ci(np.asarray(x, dtype=float))
                out.append(dict(model=m, period=P, label=lab, contrast=cname, n=len(x),
                                mean=mean, ci_lo=lo, ci_hi=hi))
    # câte serii au fost excluse: nu depinde de model, deci luăm un singur model
    one = df[df["model"] == df["model"].iloc[0]].drop_duplicates(["period", "pos", "series"])
    excl = (one.assign(up_excl=one["a_in"] < min_amp, dn_excl=one["d_in"] < min_amp)
               .groupby(["period", "pos"])
               .agg(level=("level", "mean"), up_excluse=("up_excl", "sum"),
                    drop_excluse=("dn_excl", "sum"), n=("series", "count")).reset_index())
    return pd.DataFrame(out), excl


AMP_BINS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0)


def position_r2(df, bins=AMP_BINS):
    """Cât explică poziția (vale/pantă/vârf) DUPĂ ce știm deja amplitudinea?

    Două regresii liniare pe rânduri individuale (o serie la o amplitudine):
      R² amplitudine          : echo ~ intervalul de amplitudine (variabile 0/1)
      R² amplitudine + poziție: la fel, plus eticheta poziției
    ΔR² mic = curbele se suprapun: ecoul depinde de mărime, nu de interval."""
    import pandas as pd
    out = []
    for (m, P), g in df.groupby(["model", "period"]):
        b = pd.cut(g["amp"], bins).astype(str)
        X1 = pd.get_dummies(b, dtype=float)
        X2 = pd.concat([X1, pd.get_dummies(g["label"], drop_first=True, dtype=float)], axis=1)
        y = g["echo"].values

        def r2(X):
            beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
            return 1 - ((y - X.values @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        r_a, r_ap = r2(X1), r2(X2)
        out.append(dict(model=m, period=P, r2_amp=r_a, r2_amp_poz=r_ap, delta_r2=r_ap - r_a, n=len(g)))
    return pd.DataFrame(out)
