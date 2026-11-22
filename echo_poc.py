"""
echo_poc.py — primul experiment pentru tema „ecoul anomaliilor în contextul TSFM”.

Ideea, pe scurt:
  1. Generăm o serie sezonieră curată (perioada P).
  2. Facem o copie („geamănul”) în care injectăm o anomalie în CONTEXT,
     cu k perioade înainte de originea prognozei.
  3. Prognozăm ambele variante cu același model.
  4. Diferența dintre cele două prognoze = efectul anomaliei.
     Dacă modelul „copiază” anomalia, diferența apare exact la faza unde
     anomalia ar cădea dacă s-ar repeta după k perioade. Asta e ECOUL.

Raport de ecou = media (prognoză_anomalie - prognoză_curată) pe fereastra
de ecou, împărțită la amplitudinea anomaliei.
  ≈ 1  -> modelul repetă anomalia întreagă (ca seasonal naive)
  ≈ 0  -> modelul o ignoră
"""

import numpy as np

# ----------------------------------------------------------------------------
# 1. Generatorul de serii
# ----------------------------------------------------------------------------

def make_series(n, period, noise=0.1, seed=0):
    """Serie sezonieră: sinus + o armonică + zgomot gaussian. Amplitudine ~1."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    phase = rng.uniform(0, 2 * np.pi)
    y = np.sin(2 * np.pi * t / period + phase) + 0.3 * np.sin(4 * np.pi * t / period)
    return y + noise * rng.standard_normal(n)


def inject(y, start, length, kind, amp):
    """Întoarce o COPIE a seriei cu anomalia injectată în [start, start+length).

    kind:
      'spike'   -> un singur punct ridicat (length ignorat, se folosește 1)
      'burst'   -> un bloc ridicat de `length` puncte
      'level'   -> nivel ridicat pe `length` puncte, apoi revine
      'pattern' -> forma sezonieră e inversată pe `length` puncte
                   (abaterea medie e ~0, deci raportul de ecou nu se aplică;
                    pentru acest tip folosește corelația, mai târziu)
    amp este în unități de deviație standard a seriei.
    """
    y2 = y.copy()
    sd = y.std()
    if kind == "spike":
        y2[start] += amp * sd
    elif kind == "burst":
        # formă de clopot, ca un eveniment real, nu un dreptunghi
        w = np.hanning(length + 2)[1:-1]
        y2[start:start + length] += amp * sd * w
    elif kind == "level":
        y2[start:start + length] += amp * sd
    elif kind == "pattern":
        seg = y2[start:start + length]
        y2[start:start + length] = seg.mean() - (seg - seg.mean())
    else:
        raise ValueError(kind)
    return y2


# ----------------------------------------------------------------------------
# 2. Unde cade anomalia și unde ar trebui să apară ecoul
# ----------------------------------------------------------------------------

def layout(context_len, period, k, offset, length):
    """Poziția anomaliei în context și fereastra de ecou în orizont.

    Anomalia începe la offset pași după începutul perioadei care e la
    k perioade înaintea originii. Ecoul așteptat cade în orizont la același
    offset (pentru că k perioade mai târziu faza e aceeași).
    """
    start = context_len - k * period + offset
    echo_lo = offset            # index în orizont (0 = primul pas prognozat)
    echo_hi = offset + length
    assert start >= 0, "contextul e prea scurt pentru k perioade"
    return start, echo_lo, echo_hi


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
    out_win = np.abs(d[mask]).mean()
    return in_win, in_abs, out_win


# ----------------------------------------------------------------------------
# 3. Baseline-uri (nu au nevoie de GPU)
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# 4. TSFM-uri (de rulat la tine, pe GPU)
# ----------------------------------------------------------------------------

def make_chronos(model_id="amazon/chronos-bolt-small", device="cuda", dtype="float32"):
    """Întoarce o funcție forecast(batch_ctx, h) -> mediana prognozei.

    Testat cu chronos-forecasting 2.3.2 + transformers 5.23:
      - predict_quantiles primește seria în `inputs=` (nu `context=`);
      - dtype se dă cu `dtype=` (transformers 5 a redenumit `torch_dtype`).
    float32 implicit: metrica e o DIFERENȚĂ între două prognoze aproape
    identice, iar rotunjirile din bfloat16 ar adăuga zgomot fals în `control`.
    """
    import torch
    from chronos import BaseChronosPipeline, Chronos2Pipeline

    # from_pretrained citește config-ul modelului și alege singur clasa
    # potrivită: ChronosBoltPipeline sau Chronos2Pipeline
    pipe = BaseChronosPipeline.from_pretrained(
        model_id, device_map=device, dtype=getattr(torch, dtype)
    )
    q = [0.1, 0.5, 0.9]

    def forecast_bolt(batch_ctx, h):
        # tensor 2D (lot, timp); pipeline-ul îl mută singur pe GPU
        ctx = torch.tensor(np.asarray(batch_ctx), dtype=torch.float32)
        quantiles, mean = pipe.predict_quantiles(
            inputs=ctx, prediction_length=h, quantile_levels=q
        )
        # quantiles: (lot, orizont, 3) în fp32 pe CPU; coloana 1 = mediana
        return quantiles[:, :, 1].numpy()

    def forecast_chronos2(batch_ctx, h):
        # Chronos-2 vrea 3D: (lot, variabile, timp); noi avem o singură variabilă
        ctx = torch.tensor(np.asarray(batch_ctx), dtype=torch.float32)[:, None, :]
        quantiles, mean = pipe.predict_quantiles(
            inputs=ctx, prediction_length=h, quantile_levels=q,
            cross_learning=False,  # seriile din lot NU își împrumută informație
        )
        # listă cu câte un tensor (1, orizont, 3) per serie -> (lot, orizont)
        return np.stack([qt[0, :, 1].float().cpu().numpy() for qt in quantiles])

    return forecast_chronos2 if isinstance(pipe, Chronos2Pipeline) else forecast_bolt


# ----------------------------------------------------------------------------
# 5. Experimentul
# ----------------------------------------------------------------------------

def run(forecasters, period=24, n_periods_ctx=8, horizon=None, kinds=("spike", "burst"),
        amps=(3, 5, 10), ks=(1, 2), n_series=30, offset=5, burst_len=6):
    """Întoarce o listă de rânduri (dict) cu raportul de ecou per serie."""
    horizon = horizon or period
    C = n_periods_ctx * period
    rows = []
    for kind in kinds:
        L = 1 if kind == "spike" else burst_len
        for amp in amps:
            for k in ks:
                start, lo, hi = layout(C, period, k, offset, L)
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
                    fc_c = f(clean, horizon)
                    fc_a = f(anom, horizon)
                    for i in range(n_series):
                        r_in, r_abs, r_out = echo_ratio(fc_a[i], fc_c[i], lo, hi, inj[i])
                        rows.append(dict(model=name, period=period, kind=kind,
                                         amp=amp, k=k, series=i, echo=r_in,
                                         echo_abs=r_abs, control=r_out))
    return rows


def batchify(fn, period):
    """Transformă un baseline pe o serie într-unul pe un lot de serii."""
    return lambda batch, h: np.array([fn(c, h, period) for c in batch])


def summarize(rows):
    """Media ecoului + cât din serii au ecou mare (> 0,5).

    A doua coloană deosebește două mecanisme cu aceeași medie:
      - mediere pe perioade: fiecare serie are ecou ~0,125, deci frac_high = 0
      - copiere (parroting): unele serii au ecou 1, altele 0, deci frac_high > 0
    """
    import pandas as pd
    df = pd.DataFrame(rows)
    df["high"] = df["echo"] > 0.5
    return (df.groupby(["model", "period", "kind", "k", "amp"])
              .agg(echo=("echo", "mean"), echo_abs=("echo_abs", "mean"),
                   frac_high=("high", "mean"), control=("control", "mean"))
              .round(3))


# ----------------------------------------------------------------------------
# 6. Grafic pentru o singură serie
# ----------------------------------------------------------------------------

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
    fc_c, fc_a = forecaster(np.array([clean, anom]), h)   # lot de 2 serii
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
    ax1.plot([], [], color=C_MUTED, lw=2, ls="--", label="prognoză")  # doar în legendă

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
    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))     # loc jos pentru legendă
    out = out or f"figures/one_{name}_{kind}_k{k}_amp{amp}_s{seed}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, r_in, r_abs, r_out


# Modelele Chronos disponibile: nume scurt -> (nume în tabel, ID Hugging Face)
CHRONOS_MODELS = {
    "small": ("chronos_bolt_small", "amazon/chronos-bolt-small"),
    "base": ("chronos_bolt_base", "amazon/chronos-bolt-base"),
    "c2": ("chronos_2", "amazon/chronos-2"),
}


def arg(name, default=None):
    """Valoarea de după `name` în linia de comandă (ex. --seed 3), altfel default.

    Dacă opțiunea există dar nu are valoare (ex. `--chronos` singur), întoarce ""."""
    import sys
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    nxt = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
    return "" if nxt.startswith("--") else nxt


if __name__ == "__main__":
    # Exemple:
    #   python echo_poc.py                          doar baseline-urile
    #   python echo_poc.py --chronos                + toate modelele Chronos
    #   python echo_poc.py --chronos small,c2       + doar cele alese
    #   python echo_poc.py --plot --chronos small --kind burst --amp 3 --seed 3
    #   python echo_poc.py --grid --chronos         grila P=24 și 48 -> results/grid.csv
    import sys

    def baselines(P):
        """Baseline-urile depind de perioadă, deci le refacem pentru fiecare P."""
        return {
            "seasonal_naive": batchify(seasonal_naive, P),
            "seasonal_mean": batchify(seasonal_mean, P),
            "seasonal_median": batchify(seasonal_median, P),
            "context_parrot": lambda batch, h: np.array([context_parrot(c, h) for c in batch]),
        }

    # Modelele Chronos nu depind de perioadă: le încărcăm o singură dată
    tsfm = {}
    chosen = arg("--chronos")
    if chosen is not None:
        keys = chosen.split(",") if chosen else list(CHRONOS_MODELS)
        for key in keys:
            name, model_id = CHRONOS_MODELS[key]
            tsfm[name] = make_chronos(model_id)

    if "--grid" in sys.argv:
        import os
        import pandas as pd
        rows = []
        for P in (24, 48):
            rows += run({**baselines(P), **tsfm}, period=P)
            print(f"P={P}: gata")
        os.makedirs("results", exist_ok=True)
        out = arg("--out") or "results/grid.csv"
        pd.DataFrame(rows).to_csv(out, index=False)   # un rând per serie
        print(f"{len(rows)} rânduri salvate în {out}\n")
        print(summarize(rows).to_string())
        sys.exit(0)

    P = 24
    forecasters = {**baselines(P), **tsfm}
    if "--plot" in sys.argv:
        # câte un grafic per model, la k=1 și k=2, apoi ne oprim
        seed = int(arg("--seed", 0))
        kind = arg("--kind", "spike")
        amp = float(arg("--amp", 5))
        amp = int(amp) if amp.is_integer() else amp   # „amp3” în numele fișierului, nu „amp3.0”
        for name, f in forecasters.items():
            for k in (1, 2):
                path, r_in, r_abs, r_out = plot_one(f, name, period=P, kind=kind,
                                                    amp=amp, k=k, seed=seed)
                print(f"{path}: ecou={r_in:.3f}, ecou_abs={r_abs:.3f}, control={r_out:.3f}")
        sys.exit(0)
    rows = run(forecasters, period=P)
    print(summarize(rows).to_string())