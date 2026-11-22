"""Toate graficele. Culorile vin din paleta validată (vezi skill-ul dataviz)."""

import numpy as np

from .data import inject, layout, make_series
from .metrics import bootstrap_ci, echo_ratio, split_q
from .plausibility import AMP_BINS, position_r2


# Culori din paleta validată (albastru / portocaliu se disting și pentru
# daltoniști); textul și liniile de referință rămân gri.
C_CLEAN, C_ANOM = "#2a78d6", "#eb6834"


C_INK, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_one(forecaster, name, period=24, n_periods_ctx=8, kind="spike", amp=5,
             k=1, offset=5, burst_len=6, seed=0, show_periods=3, out=None):
    """O serie curată vs geamănul ei cu anomalie + efectul anomaliei în prognoză.

    Sus: contextul (linie plină) și prognoza (întreruptă) pentru ambele variante.
    Jos: diferența de prognoză / abaterea injectată = ce mediază echo_ratio().
    """
    import os
    import matplotlib
    matplotlib.use("Agg")            # salvăm direct în fișier, fără fereastră
    import matplotlib.pyplot as plt

    # [A] Datele: exact aceleași funcții ca în run(), pentru o singură serie
    C, h = n_periods_ctx * period, period
    L = 1 if kind == "spike" else burst_len
    start, lo, hi = layout(C, period, k, offset, L)
    clean = make_series(C, period, seed=seed)
    anom = inject(clean, start, L, kind, amp)
    inj = (anom[start:start + L] - clean[start:start + L]).mean()
    med, q10, q90 = split_q(forecaster(np.array([clean, anom]), h))  # lot de 2
    fc_c, fc_a = med
    r_in, r_abs, r_out = echo_ratio(fc_a, fc_c, lo, hi, inj)

    # [B] Axa timpului: 0..C-1 = context, C..C+h-1 = orizont.
    #     Arătăm doar ultimele perioade din context (minim k+1, ca să se vadă anomalia).
    t_ctx, t_fc = np.arange(C), np.arange(C, C + h)
    x0 = C - max(show_periods, k + 1) * period

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 6.5),
                                   gridspec_kw=dict(height_ratios=[2.2, 1]))

    # [C] Panoul de sus: seriile și prognozele. Portocaliul e desenat primul,
    #     ca albastrul să-l acopere acolo unde cele două coincid.
    ax1.plot(t_ctx[x0:], anom[x0:], color=C_ANOM, lw=2, label="seria cu anomalie")
    ax1.plot(t_ctx[x0:], clean[x0:], color=C_CLEAN, lw=2, label="seria curată")
    ax1.plot(t_fc, fc_a, color=C_ANOM, lw=2, ls="--")
    ax1.plot(t_fc, fc_c, color=C_CLEAN, lw=2, ls="--")
    ax1.plot([], [], color=C_MUTED, lw=2, ls="--", label="prognoză (mediana)")  # doar în legendă
    if q10 is not None:
        # banda q0.1–q0.9: cât de sigur e modelul (doar TSFM-urile o au)
        for i, color in ((1, C_ANOM), (0, C_CLEAN)):
            ax1.fill_between(t_fc, q10[i], q90[i], color=color, alpha=0.18, lw=0)
        ax1.fill_between([], [], [], color=C_MUTED, alpha=0.25, lw=0,
                         label="interval q0,1–q0,9")

    # [D] Marcaje: banda anomaliei, fereastra de ecou (hașurată), originea prognozei
    ax1.axvspan(start - 0.5, start + L - 0.5, color=C_ANOM, alpha=0.15, lw=0)
    for ax in (ax1, ax2):
        ax.axvspan(C + lo - 0.5, C + hi - 0.5, facecolor="none", edgecolor=C_MUTED,
                   hatch="///", lw=0, alpha=0.5)
        ax.axvline(C - 0.5, color=C_MUTED, lw=1)
    ytop, ybot = ax1.get_ylim()[1], ax1.get_ylim()[0]
    note = dict(textcoords="offset points", color=C_MUTED, fontsize=9)
    ax1.annotate("anomalie", (start, ytop), xytext=(0, -2), ha="center", va="top", **note)
    ax1.annotate("fereastra de ecou", (C + (lo + hi) / 2, ytop), xytext=(0, -2),
                 ha="center", va="top", **note)
    ax1.annotate("originea prognozei", (C - 0.5, ybot), xytext=(-4, 4),
                 ha="right", va="bottom", **note)
    ax1.set_ylabel("valoare")
    ax1.set_title(f"{name}  |  {kind}, {amp}σ, k={k}, P={period}, serie {seed}  |  "
                  f"ecou = {r_in:.2f}, ecou_abs = {r_abs:.2f}, control = {r_out:.2f}",
                  loc="left", fontsize=10, color=C_INK)

    # [E] Panoul de jos: diferența normalizată, citită direct ca raport de ecou
    d = (fc_a - fc_c) / inj
    for y, lbl in ((1, "copiere completă (1)"), (0, "nicio urmă (0)")):
        ax2.axhline(y, color=C_MUTED, lw=1, ls=":")
        ax2.annotate(lbl, (x0, y), xytext=(2, 2), textcoords="offset points",
                     fontsize=8, color=C_MUTED)
    ax2.plot(t_fc, d, color=C_INK, lw=2, marker="o", ms=4)
    ax2.set_ylabel("Δ prognoză /\nabatere injectată", fontsize=9)
    ax2.set_xlabel("pas de timp")
    ax2.set_ylim(min(-0.2, d.min() - 0.1), max(1.2, d.max() + 0.1))
    ax2.set_xlim(x0, C + h)

    # [F] Stil: axe și grilă discrete, legenda sub figură, salvare PNG
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(C_MUTED)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        ax.grid(axis="y", color=C_GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", ncol=4,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))     # loc jos pentru legendă
    out = out or f"figures/one_{name}_P{period}_{kind}L{L}_k{k}_amp{amp}_s{seed}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, r_in, r_abs, r_out


# Modelele afișate colorat (sloturile 1-3 din paleta validată + forme diferite,
# ca identitatea să nu depindă doar de culoare)
TSFM_STYLE = {
    "chronos_bolt_small": ("Chronos-Bolt small", "#2a78d6", "o"),
    "chronos_bolt_base": ("Chronos-Bolt base", "#eb6834", "s"),
    "chronos_2": ("Chronos-2", "#1baf7a", "^"),
    "timesfm_2p5": ("TimesFM 2.5", "#eda100", "v"),
    "tirex": ("TiRex", "#e87ba4", "P"),
    "chronos_t5_small": ("Chronos T5 small", "#008300", "X"),
    "moirai_2_small": ("Moirai 2.0 small", "#6250d6", "*"),
}


H1_THRESHOLD = 0.3


def plot_main(csv="results/grid.csv", k=1, metric="echo", out=None):
    """Ecoul mediu (± IC bootstrap 95%) per model, pe perioadă × tip × amplitudine.

    Rânduri = tip de anomalie, coloane = perioadă, axa x = amplitudinea.
    Baseline-urile apar ca referințe gri, pragul H1 ca linie întreruptă.
    Întoarce (calea figurii, tabelul cu medii și intervale)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    # [A] Date: media și IC per (model, perioadă, tip, amplitudine) la k ales
    df = pd.read_csv(csv)
    df = df[df["k"] == k]
    rows = []
    for (model, P, kind, amp), g in df.groupby(["model", "period", "kind", "amp"]):
        lo, hi = bootstrap_ci(g[metric].values)
        rows.append(dict(model=model, period=P, kind=kind, amp=amp, k=k,
                         mean=g[metric].mean(), ci_lo=lo, ci_hi=hi, n=len(g)))
    tab = pd.DataFrame(rows)

    periods = sorted(tab["period"].unique())
    kinds = [kd for kd in ("burst", "spike") if kd in set(tab["kind"])]
    amps = sorted(tab["amp"].unique())
    kind_ro = {"burst": "rafală", "spike": "vârf"}

    fig, axes = plt.subplots(len(kinds), len(periods), figsize=(10, 3.6 * len(kinds)),
                             sharey=True, squeeze=False)
    # pozițiile pe x: câte un grup per amplitudine, modelele decalate în grup
    names = list(TSFM_STYLE) + ["context_parrot"]
    dx = np.linspace(-0.27, 0.27, len(names))

    for r, kind in enumerate(kinds):
        for c, P in enumerate(periods):
            ax = axes[r, c]
            sub = tab[(tab["kind"] == kind) & (tab["period"] == P)]

            # [B] Referințe: pragul H1 și media pe faze (constantă 1/8 = 0.125)
            ax.axhline(H1_THRESHOLD, color=C_INK, lw=1.2, ls="--")
            ax.axhline(0.125, color=C_MUTED, lw=1, ls=":")
            ax.axhline(0, color=C_MUTED, lw=0.8)

            # [C] Mediana pe faze: segment scurt gri în fiecare grup
            #     (depinde de amplitudine, dar rămâne aproape de 0)
            for j, amp in enumerate(amps):
                m = sub[(sub["model"] == "seasonal_median") & (sub["amp"] == amp)]
                if len(m):
                    ax.hlines(m["mean"].iloc[0], j - 0.4, j + 0.4, color=C_MUTED,
                              lw=1.5, label="mediana pe faze" if j == 0 else None)

            # [D] Puncte + IC: modelele TSFM colorate, context_parrot gri
            for i, name in enumerate(names):
                if name in TSFM_STYLE:
                    label, color, marker = TSFM_STYLE[name]
                else:
                    label, color, marker = "context parroting", C_MUTED, "D"
                for j, amp in enumerate(amps):
                    m = sub[(sub["model"] == name) & (sub["amp"] == amp)]
                    if not len(m):
                        continue
                    m = m.iloc[0]
                    x = j + dx[i]
                    ax.vlines(x, m["ci_lo"], m["ci_hi"], color=color, lw=2)
                    ax.plot(x, m["mean"], marker=marker, ms=7, color=color,
                            mec="white", mew=1, ls="none",
                            label=label if j == 0 else None)

            # [E] Axe și titluri
            ax.set_xticks(range(len(amps)))
            ax.set_xticklabels([f"{a:g}σ" for a in amps])
            ax.set_xlim(-0.5, len(amps) - 0.5)
            ax.set_title(f"{kind_ro[kind]}, perioadă {P}", fontsize=10, color=C_INK,
                         loc="left")
            if c == 0:
                ax.set_ylabel(f"{metric} (k={k})", fontsize=9)
            if r == len(kinds) - 1:
                ax.set_xlabel("amplitudinea anomaliei", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(C_MUTED)
            ax.tick_params(colors=C_MUTED, labelsize=8)
            ax.grid(axis="y", color=C_GRID, lw=0.8)
            ax.set_axisbelow(True)

    lo = min(-0.02, tab[tab["model"].isin(names)]["ci_lo"].min() - 0.01)
    axes[0, 0].set_ylim(lo, 0.36)

    # [F] Titlu, notă despre seasonal naive (=1, în afara axei), legendă jos
    fig.suptitle(f"Ecoul anomaliei la k={k} perioade înainte de origine "
                 f"(media pe 30 de serii, IC bootstrap 95%)", fontsize=11,
                 color=C_INK, x=0.01, ha="left")
    fig.text(0.01, 0.935, "Referințe: seasonal naive = 1 (copie completă, deasupra axei); "
             "media pe faze = 0,125; mediana pe faze ≈ 0.", fontsize=8, color=C_MUTED)
    # liniile de referință intră în legendă (etichetele în grafic s-ar suprapune)
    from matplotlib.lines import Line2D
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [Line2D([], [], color=C_INK, lw=1.2, ls="--"),
                Line2D([], [], color=C_MUTED, lw=1, ls=":")]
    labels += ["prag H1 (0,3)", "media pe faze (0,125)"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    out = out or f"figures/main_{metric}_k{k}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, tab


def plot_by_length(csv="results/grid2.csv", k=1, metric="echo", out=None):
    """Grafic pentru grila proporțională: metrica pe lungimea anomaliei.

    Coloane = perioadă, rânduri = amplitudine, axa x = tipul/lungimea anomaliei.
    metric="echo": media ± IC bootstrap, cu referințele de la baseline-uri.
    metric="width_in"/"width_out": raport de lățimi -> media GEOMETRICĂ ± IC,
    pe axă logaritmică (un raport de 2 și unul de 0,5 sunt la fel de departe de 1)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D

    is_ratio = metric in ("width_in", "width_out")      # rapoarte -> axă log
    is_abs = metric.startswith("width_abs")              # diferențe -> axă liniară
    df = pd.read_csv(csv)
    df = df[df["k"] == k].copy()
    df["cfg"] = df["kind"] + "_" + df["len_frac"].astype(str)
    cfg_order = [("burst_0.25", "rafală P/4"), ("burst_0.5", "rafală P/2"),
                 ("burst_1.0", "rafală P"), ("level_1.0", "level P")]
    periods, amps = sorted(df["period"].unique()), sorted(df["amp"].unique())

    # [A] Media și IC per (model, P, amp, cfg); la rapoarte lucrăm pe log
    rows = []
    for (model, P, amp, cfg), g in df.groupby(["model", "period", "amp", "cfg"]):
        x = g[metric].dropna().values
        if not len(x):
            continue                       # baseline-urile n-au interval
        if is_ratio:
            lo, hi = bootstrap_ci(np.log(x))
            m, lo, hi = np.exp(np.log(x).mean()), np.exp(lo), np.exp(hi)
        else:
            (lo, hi), m = bootstrap_ci(x), x.mean()
        rows.append(dict(model=model, period=P, amp=amp, cfg=cfg, k=k,
                         metric=metric, mean=m, ci_lo=lo, ci_hi=hi, n=len(x)))
    tab = pd.DataFrame(rows)

    names = [m for m in TSFM_STYLE if m in set(tab["model"])]
    if not (is_ratio or is_abs):
        names.append("context_parrot")
    dx = np.linspace(-0.33, 0.33, len(names))
    fig, axes = plt.subplots(len(amps), len(periods), figsize=(10, 2.9 * len(amps)),
                             sharey=True, sharex=True, squeeze=False)

    for r, amp in enumerate(amps):
        for c, P in enumerate(periods):
            ax = axes[r, c]
            sub = tab[(tab["period"] == P) & (tab["amp"] == amp)]
            # [B] Referințe
            if is_ratio:
                ax.axhline(1, color=C_MUTED, lw=1)
                ax.set_yscale("log")
                # etichete obișnuite (1, 2, 5...) în loc de 10^0, 10^1
                from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
                ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50]))
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.yaxis.set_minor_locator(NullLocator())
            elif is_abs:
                ax.axhline(0, color=C_MUTED, lw=1)
                ax.axhline(1, color=C_MUTED, lw=1, ls=":")
            else:
                ax.axhline(H1_THRESHOLD, color=C_INK, lw=1.2, ls="--")
                ax.axhline(1, color=C_MUTED, lw=1, ls="-.")
                ax.axhline(0.125, color=C_MUTED, lw=1, ls=":")
                ax.axhline(0, color=C_MUTED, lw=0.8)
            # [C] Puncte + IC
            for i, name in enumerate(names):
                label, color, marker = TSFM_STYLE.get(name, ("context parroting", C_MUTED, "D"))
                for j, (cfg, _) in enumerate(cfg_order):
                    m = sub[(sub["model"] == name) & (sub["cfg"] == cfg)]
                    if not len(m):
                        continue
                    m = m.iloc[0]
                    ax.vlines(j + dx[i], m["ci_lo"], m["ci_hi"], color=color, lw=2)
                    ax.plot(j + dx[i], m["mean"], marker=marker, ms=7, color=color,
                            mec="white", mew=1, ls="none", label=label if j == 0 else None)
            # [D] Axe
            ax.set_xticks(range(len(cfg_order)))
            ax.set_xticklabels([lbl for _, lbl in cfg_order])
            ax.set_xlim(-0.5, len(cfg_order) - 0.5)
            ax.set_title(f"perioadă {P}, amplitudine {amp:g}σ", fontsize=10,
                         color=C_INK, loc="left")
            if c == 0:
                ax.set_ylabel(metric, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(C_MUTED)
            ax.tick_params(colors=C_MUTED, labelsize=8)
            ax.grid(axis="y", color=C_GRID, lw=0.8)
            ax.set_axisbelow(True)

    # [E] Titlu, legendă (cu liniile de referință), salvare
    what = {"echo": "Ecoul (mediana prognozei)",
            "echo_q90": "Ecoul pe cuantila 0,9 a prognozei",
            "width_in": "Lățimea intervalului q0,1–q0,9 în fereastra de ecou, față de geamănul curat",
            "width_out": "Lățimea intervalului q0,1–q0,9 în afara ferestrei, față de geamănul curat",
            "width_abs_in": "Creșterea lățimii intervalului q0,1–q0,9 în fereastra de ecou, "
                            "/ amplitudinea anomaliei",
            "width_abs_out": "Creșterea lățimii intervalului q0,1–q0,9 în afara ferestrei, "
                             "/ amplitudinea anomaliei"}
    stat = "media geometrică" if is_ratio else "media"
    fig.suptitle(f"{what.get(metric, metric)}, k={k} ({stat} pe serii, IC bootstrap 95%)",
                 fontsize=11, color=C_INK, x=0.01, ha="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if is_ratio:
        handles.append(Line2D([], [], color=C_MUTED, lw=1)); labels.append("neschimbat (1)")
    elif is_abs:
        handles += [Line2D([], [], color=C_MUTED, lw=1), Line2D([], [], color=C_MUTED, lw=1, ls=":")]
        labels += ["neschimbat (0)", "lărgire = amplitudinea anomaliei (1)"]
    else:
        handles += [Line2D([], [], color=C_INK, lw=1.2, ls="--"),
                    Line2D([], [], color=C_MUTED, lw=1, ls="-."),
                    Line2D([], [], color=C_MUTED, lw=1, ls=":")]
        labels += ["prag H1 (0,3)", "seasonal naive (1)", "media pe faze (0,125)"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out = out or f"figures/bylen_{metric}_k{k}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, tab


def plot_persistence(curves, out="figures/persist_dh.png"):
    """Curba medie d(h) pentru fiecare gap: unde cade efectul în orizont?

    Rânduri = modele, coloane = perioade. Fiecare gap are o culoare (de la
    deschis = anomalie lipită de origine, la închis = departe de origine);
    bara de aceeași culoare de sus arată fereastra de fază a acelui gap."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    cv = pd.DataFrame(curves)
    models = [m for m in TSFM_STYLE if m in set(cv["model"])]
    periods = sorted(cv["period"].unique())
    # rampă secvențială albastră (o singură nuanță, deschis -> închis)
    ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    fig, axes = plt.subplots(len(models), len(periods), figsize=(11, 2.6 * len(models)),
                             sharey=True, squeeze=False)
    for r, m in enumerate(models):
        for c, P in enumerate(periods):
            ax = axes[r, c]
            sub = cv[(cv["model"] == m) & (cv["period"] == P)]
            gaps = sorted(sub["gap"].unique())
            ax.axhline(0, color=C_MUTED, lw=0.8)
            ax.axhline(1, color=C_MUTED, lw=1, ls=":")
            for j, gap in enumerate(gaps):
                g = sub[sub["gap"] == gap].sort_values("h")
                col = ramp[j]
                ax.fill_between(g["h"], g["d_mean"] - 1.96 * g["d_se"],
                                g["d_mean"] + 1.96 * g["d_se"], color=col, alpha=0.2, lw=0)
                ax.plot(g["h"], g["d_mean"], color=col, lw=2,
                        label=["gap 0 (lipită de origine)", "gap P/8", "gap P/4", "gap P/2"][j])
                off, L = g["offset"].iloc[0], g["L"].iloc[0]
                yb = 2.05 - 0.09 * j
                ax.hlines(yb, off - 0.5, off + L - 0.5, color=col, lw=4)
            ax.set_ylim(-0.3, 2.15)
            ax.set_xlim(-0.5, P - 0.5)
            ax.set_title(f"{TSFM_STYLE[m][0]}, perioadă {P}", fontsize=10, color=C_INK, loc="left")
            if c == 0:
                ax.set_ylabel("d(h) mediu", fontsize=9)
            if r == len(models) - 1:
                ax.set_xlabel("pas în orizont (0 = imediat după origine)", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(C_MUTED)
            ax.tick_params(colors=C_MUTED, labelsize=8)
            ax.grid(axis="y", color=C_GRID, lw=0.8)
            ax.set_axisbelow(True)
    fig.suptitle("Persistență sau copiere de fază? Rafală P/2, 5σ, k=1, cu distanță variabilă "
                 "până la origine (medie pe 100 de serii ± IC 95%)",
                 fontsize=11, color=C_INK, x=0.01, ha="left")
    fig.text(0.01, 0.935, "Barele de sus = fereastra de fază a fiecărui gap. Copierea de fază: "
             "curba urmează bara. Persistența: curba rămâne la stânga și scade cu gap-ul.",
             fontsize=8, color=C_MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_plausibility(df, min_amp=0.5, out_pos="figures/plaus_positions.png",
                      out_dose="figures/plaus_dose.png"):
    """(a) echo pe cele 8 puncte ale ciclului: in vs out, rafală în sus și drop.
    (b) doză–răspuns: echo față de multiplul lui a_in (linia verticală = pragul)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in TSFM_STYLE if m in set(df["model"])]
    periods = sorted(df["period"].unique())
    up = df[df["a_in"] >= min_amp]
    dn = df[df["d_in"] >= min_amp]

    def style(ax, r, c, xlabel):
        ax.axhline(0, color=C_MUTED, lw=0.8)
        ax.axhline(H1_THRESHOLD, color=C_INK, lw=1, ls="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(C_MUTED)
        ax.tick_params(colors=C_MUTED, labelsize=7)
        ax.grid(axis="y", color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
        if c == 0:
            ax.set_ylabel(f"echo, P={periods[r]}", fontsize=9)
        if r == len(periods) - 1:
            ax.set_xlabel(xlabel, fontsize=8)

    # (a) pozițiile
    fig, axes = plt.subplots(len(periods), len(models), figsize=(3.1 * len(models), 3.2 * len(periods)),
                             sharey=True, squeeze=False)
    series_def = [("in_range", up, C_CLEAN, "-", "în sus, în interval"),
                  ("out_range", up, C_ANOM, "-", "în sus, peste interval"),
                  ("drop_in", dn, C_CLEAN, "--", "drop, în interval"),
                  ("drop_out", dn, C_ANOM, "--", "drop, sub interval")]
    for r, P in enumerate(periods):
        for c, m in enumerate(models):
            ax = axes[r, c]
            for vname, src, col, ls, lbl in series_def:
                g = src[(src["model"] == m) & (src["period"] == P) & (src["variant"] == vname)]
                s = g.groupby("pos")["echo"].agg(["mean", "sem"])
                ax.fill_between(s.index, s["mean"] - 1.96 * s["sem"], s["mean"] + 1.96 * s["sem"],
                                color=col, alpha=0.15, lw=0)
                ax.plot(s.index, s["mean"], color=col, ls=ls, lw=2, marker="o", ms=3, label=lbl)
            lv = df[(df["period"] == P) & (df["model"] == m)].groupby("pos")["level"].mean()
            ax.set_xticks(lv.index)
            ax.set_xticklabels([f"{j}\n{v:.1f}" for j, v in lv.items()])
            ax.set_title(TSFM_STYLE[m][0], fontsize=9, color=C_INK, loc="left")
            style(ax, r, c, "punct în ciclu\n(nivel: 0 = vale, 1 = vârf)")
    fig.suptitle("Ecoul după poziția anomaliei în ciclu: în interval vs. peste interval "
                 "(rafală P/2, k=1, medie ± IC 95%)", fontsize=11, color=C_INK, x=0.01, ha="left")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    os.makedirs(os.path.dirname(out_pos) or ".", exist_ok=True)
    fig.savefig(out_pos, dpi=150)
    plt.close(fig)

    # (b) doză–răspuns
    ramp = {"vale": "#86b6ef", "pantă": "#2a78d6", "vârf": "#0d366b"}
    fig, axes = plt.subplots(len(periods), len(models), figsize=(3.1 * len(models), 3.2 * len(periods)),
                             sharey=True, squeeze=False)
    sw = up[up["variant"] == "sweep"]
    for r, P in enumerate(periods):
        for c, m in enumerate(models):
            ax = axes[r, c]
            ax.axvline(1, color=C_MUTED, lw=1, ls=":")
            for lab, col in ramp.items():
                g = sw[(sw["model"] == m) & (sw["period"] == P) & (sw["label"] == lab)]
                if not len(g):
                    continue
                s = g.groupby("mult")["echo"].agg(["mean", "sem"])
                ax.fill_between(s.index, s["mean"] - 1.96 * s["sem"], s["mean"] + 1.96 * s["sem"],
                                color=col, alpha=0.15, lw=0)
                ax.plot(s.index, s["mean"], color=col, lw=2, marker="o", ms=3, label=lab)
            ax.set_xscale("log")
            from matplotlib.ticker import NullLocator
            ax.xaxis.set_minor_locator(NullLocator())   # altfel apar etichete „3×10⁻¹” suprapuse
            ax.set_xticks([0.25, 0.5, 1, 2, 3])
            ax.set_xticklabels(["0,25", "0,5", "1", "2", "3"])
            ax.set_title(TSFM_STYLE[m][0], fontsize=9, color=C_INK, loc="left")
            style(ax, r, c, "amplitudine / a_in\n(1 = atinge maximul seriei)")
    fig.suptitle("Doză–răspuns: ecoul față de amplitudine, raportată la pragul intervalului "
                 "(rafală P/2, k=1, medie ± IC 95%)", fontsize=11, color=C_INK, x=0.01, ha="left")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9, title="poziția anomaliei",
               title_fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out_dose, dpi=150)
    plt.close(fig)
    return out_pos, out_dose


def plot_dose_abs(df, out="figures/plaus_dose_abs.png", bins=AMP_BINS, min_n=10):
    """Doză–răspuns pe amplitudinea ABSOLUTĂ (σ), toate pozițiile pe același panou.

    Folosește toate rafalele în sus din plausibility.csv (sweep, in_range,
    out_range). Amplitudinile se grupează în intervale; un punct = media
    ecoului dintr-un interval (minim min_n serii), cu IC 95%."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.ticker import NullLocator

    up = df[(df["direction"] > 0) & (df["amp"] >= bins[0]) & np.isfinite(df["echo"])].copy()
    up["bin"] = pd.cut(up["amp"], bins)
    r2 = position_r2(up, bins)
    models = [m for m in TSFM_STYLE if m in set(up["model"])]
    periods = sorted(up["period"].unique())
    ramp = {"vale": "#86b6ef", "pantă": "#2a78d6", "vârf": "#0d366b"}
    fig, axes = plt.subplots(len(periods), len(models), figsize=(3.1 * len(models), 3.2 * len(periods)),
                             sharey=True, squeeze=False)
    for r, P in enumerate(periods):
        for c, m in enumerate(models):
            ax = axes[r, c]
            g = up[(up["model"] == m) & (up["period"] == P)]
            for lab, col in ramp.items():
                s = (g[g["label"] == lab].groupby("bin", observed=True)
                     .agg(x=("amp", lambda a: np.exp(np.log(a).mean())),
                          mean=("echo", "mean"), sem=("echo", "sem"), n=("echo", "size")))
                s = s[s["n"] >= min_n]
                ax.fill_between(s["x"], s["mean"] - 1.96 * s["sem"], s["mean"] + 1.96 * s["sem"],
                                color=col, alpha=0.15, lw=0)
                ax.plot(s["x"], s["mean"], color=col, lw=2, marker="o", ms=3, label=lab)
            q = r2[(r2["model"] == m) & (r2["period"] == P)].iloc[0]
            ax.text(0.97, 0.97, f"ΔR² poziție = {q['delta_r2']:.3f}\nR² amplitudine = {q['r2_amp']:.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=C_INK)
            ax.axhline(0, color=C_MUTED, lw=0.8)
            ax.axhline(H1_THRESHOLD, color=C_INK, lw=1, ls="--")
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
            ax.set_xticklabels(["0,25", "0,5", "1", "2", "4", "8"])
            ax.set_title(TSFM_STYLE[m][0], fontsize=9, color=C_INK, loc="left")
            if c == 0:
                ax.set_ylabel(f"echo, P={P}", fontsize=9)
            if r == len(periods) - 1:
                ax.set_xlabel("amplitudinea anomaliei (σ)", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(C_MUTED)
            ax.tick_params(colors=C_MUTED, labelsize=7)
            ax.grid(axis="y", color=C_GRID, lw=0.6)
            ax.set_axisbelow(True)
    fig.suptitle("Doză–răspuns pe amplitudinea absolută: se suprapun curbele pozițiilor? "
                 "(rafală P/2 în sus, k=1, medie ± IC 95%)", fontsize=11, color=C_INK, x=0.01, ha="left")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9,
               title="poziția anomaliei", title_fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, r2


def plot_exceed(df, out="figures/exceed_echo.png", len_frac=0.5, k=1):
    """Ecoul față de exceed_frac, pe rafala de P/2 la k=1.

    Rânduri = perioade, coloane = modele; culoarea = amplitudinea (o singură
    nuanță, deschis -> închis). În colț: Spearman pe toate punctele și
    media Spearman-ului calculat separat pe fiecare amplitudine."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[(df["kind"] == "burst") & (df["len_frac"] == len_frac) & (df["k"] == k)]
    models = [m for m in TSFM_STYLE if m in set(sub["model"])]
    periods, amps = sorted(sub["period"].unique()), sorted(sub["amp"].unique())
    ramp = ["#86b6ef", "#2a78d6", "#0d366b"]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(periods), len(models), figsize=(3 * len(models), 3.2 * len(periods)),
                             sharex=True, sharey=True, squeeze=False)
    for r, P in enumerate(periods):
        for c, m in enumerate(models):
            ax = axes[r, c]
            g = sub[(sub["period"] == P) & (sub["model"] == m)]
            for j, amp in enumerate(amps):
                ga = g[g["amp"] == amp]
                jitter = rng.uniform(-0.012, 0.012, len(ga))   # exceed_frac e discret
                ax.scatter(ga["exceed_frac"] + jitter, ga["echo"], s=10, color=ramp[j],
                           alpha=0.7, lw=0, label=f"{amp:g}σ")
            rho_all = g["exceed_frac"].corr(g["echo"], method="spearman")
            rho_amp = np.nanmean([ga["exceed_frac"].corr(ga["echo"], method="spearman")
                                  for _, ga in g.groupby("amp")])
            ax.text(0.98, 0.97, f"ρ toate = {rho_all:.2f}\nρ pe amp. = {rho_amp:.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color=C_INK)
            ax.axhline(0, color=C_MUTED, lw=0.8)
            ax.axhline(H1_THRESHOLD, color=C_INK, lw=1, ls="--")
            ax.set_title(f"{TSFM_STYLE[m][0]}, P={P}", fontsize=9, color=C_INK, loc="left")
            if c == 0:
                ax.set_ylabel("echo", fontsize=9)
            if r == len(periods) - 1:
                ax.set_xlabel("exceed_frac", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(C_MUTED)
            ax.tick_params(colors=C_MUTED, labelsize=8)
            ax.grid(color=C_GRID, lw=0.6)
            ax.set_axisbelow(True)
    fig.suptitle("Ecou față de fracțiunea anomaliei aflată în afara intervalului seriei curate "
                 "(rafală P/2, k=1; un punct = o serie)", fontsize=11, color=C_INK, x=0.01, ha="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               title="amplitudine", title_fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
