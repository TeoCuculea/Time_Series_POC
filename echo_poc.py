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
    """Întoarce o funcție forecast(batch_ctx, h) -> cuantile (lot, orizont, 3).

    Ultima axă = cuantilele [0.1, 0.5, 0.9]; split_q() le desparte.

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
        # quantiles: (lot, orizont, 3) în fp32 pe CPU: q0.1, mediana, q0.9
        return quantiles.numpy()

    def forecast_chronos2(batch_ctx, h):
        # Chronos-2 vrea 3D: (lot, variabile, timp); noi avem o singură variabilă
        ctx = torch.tensor(np.asarray(batch_ctx), dtype=torch.float32)[:, None, :]
        quantiles, mean = pipe.predict_quantiles(
            inputs=ctx, prediction_length=h, quantile_levels=q,
            cross_learning=False,  # seriile din lot NU își împrumută informație
        )
        # listă cu câte un tensor (1, orizont, 3) per serie -> (lot, orizont, 3)
        return np.stack([qt[0].float().cpu().numpy() for qt in quantiles])

    return forecast_chronos2 if isinstance(pipe, Chronos2Pipeline) else forecast_bolt


def make_timesfm(model_id="google/timesfm-2.5-200m-pytorch", max_context=512, max_horizon=128):
    """TimesFM 2.5 (Google), cu aceeași ieșire ca make_chronos: (lot, orizont, 3).

    Testat cu timesfm 3.0.2. Setările din compile() sunt cele recomandate în
    README. torch_compile=False: torch.compile cere Triton, care lipsește pe
    Windows (iar fără el modelul doar rulează puțin mai încet).
    TimesFM întoarce 10 coloane: 0 = media, 1..9 = cuantilele 0.1..0.9."""
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id, torch_compile=False)
    model.compile(timesfm.ForecastConfig(
        max_context=max_context,       # contextul nostru: max 8 × 48 = 384 de pași
        max_horizon=max_horizon,       # rotunjit la multiplu de 128 de bibliotecă
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,        # nu contează: seriile noastre au și valori negative
        fix_quantile_crossing=True,
        per_core_batch_size=64,
    ))

    def forecast(batch_ctx, h):
        _, q = model.forecast(horizon=h, inputs=[np.asarray(c, dtype=np.float32) for c in batch_ctx])
        return np.asarray(q)[:, :h][..., [1, 5, 9]]     # q0.1, mediana, q0.9

    return forecast


def make_tirex(model_id="NX-AI/TiRex", device="cuda"):
    """TiRex (NX-AI, arhitectură xLSTM, nu transformer), ieșire (lot, orizont, 3).

    Testat cu tirex-ts 1.4.2. backend="torch": kernelurile CUDA dedicate cer
    pachetul xlstm, greu de compilat pe Windows; varianta torch dă aceleași
    rezultate, doar mai lent."""
    import torch
    from tirex import load_model
    model = load_model(model_id, device=device, backend="torch")
    levels = list(model.config.quantiles)
    idx = [levels.index(0.1), levels.index(0.5), levels.index(0.9)]

    def forecast(batch_ctx, h):
        ctx = torch.tensor(np.asarray(batch_ctx), dtype=torch.float32)
        q, _ = model.forecast(context=ctx, prediction_length=h, output_type="numpy")
        return np.asarray(q)[..., idx]

    return forecast


def split_q(fc):
    """Desparte ieșirea unui model în (mediană, q0.1, q0.9).

    Baseline-urile dau o singură valoare per pas (array 2D), deci n-au
    interval: atunci q0.1 și q0.9 sunt None. TSFM-urile dau 3D (..., 3)."""
    fc = np.asarray(fc)
    if fc.ndim == 3:
        return fc[..., 1], fc[..., 0], fc[..., 2]
    return fc, None, None


# ----------------------------------------------------------------------------
# 5. Experimentul
# ----------------------------------------------------------------------------

def proportional_configs(period, kinds_fracs, offset_frac=0.2):
    """Anomalii cu lungimea ca FRACȚIUNE din perioadă: (kind, frac) -> (kind, L, offset).

    Offset-ul e tot proporțional (P/5), dar îl micșorăm dacă anomalia n-ar
    mai încăpea în perioadă: la k=1 ar trece de originea prognozei și ar fi
    tăiată fără nicio eroare. La L = P, offset-ul devine 0."""
    out = []
    for kind, frac in kinds_fracs:
        L = max(1, round(frac * period))
        off = min(round(offset_frac * period), period - L)
        out.append((kind, L, off))
    return out


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
    keys = [c for c in ["model", "period", "kind", "L", "k", "amp"] if c in df]
    agg = dict(echo=("echo", "mean"), echo_abs=("echo_abs", "mean"),
               frac_high=("high", "mean"), control=("control", "mean"))
    for c in ("echo_q90", "width_in", "width_out", "width_abs_in", "width_abs_out"):
        if c in df:
            agg[c] = (c, "mean")
    return df.groupby(keys).agg(**agg).round(3)


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


# ----------------------------------------------------------------------------
# 7. Graficul principal: ecoul mediu per model, cu interval bootstrap
# ----------------------------------------------------------------------------

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


# Modelele afișate colorat (sloturile 1-3 din paleta validată + forme diferite,
# ca identitatea să nu depindă doar de culoare)
TSFM_STYLE = {
    "chronos_bolt_small": ("Chronos-Bolt small", "#2a78d6", "o"),
    "chronos_bolt_base": ("Chronos-Bolt base", "#eb6834", "s"),
    "chronos_2": ("Chronos-2", "#1baf7a", "^"),
    "timesfm_2p5": ("TimesFM 2.5", "#eda100", "v"),
    "tirex": ("TiRex", "#e87ba4", "P"),
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


# ----------------------------------------------------------------------------
# 8. Teste de control: persistență vs. copiere de fază, granițe de patch
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# 9. H3: reduce mascarea anomaliei ecoul?
# ----------------------------------------------------------------------------

def repair(y, start, L, how, period):
    """Copie a seriei în care segmentul [start, start+L) e „reparat”.

    how:
      'anomalie'    -> nimic (referința: ecoul fără intervenție)
      'masca'       -> segmentul devine NaN; modelul îl tratează ca lipsă
      'interp'      -> interpolare liniară între punctele vecine segmentului
      'sezonier'    -> segmentul e copiat din perioada anterioară
      'masca_centru50' -> NaN doar pe jumătatea centrală (detector imprecis:
                       marginile anomaliei rămân în serie)
    Atenție: la primele 4, anomalia dispare complet, deci seria reparată e
    identică cu geamănul curat reparat la fel. Ecoul rămas = costul golului."""
    y2 = y.copy()
    if how == "anomalie":
        return y2
    if how == "masca":
        y2[start:start + L] = np.nan
    elif how == "interp":
        a, b = y2[start - 1], y2[start + L] if start + L < len(y2) else y2[start - 1]
        y2[start:start + L] = np.linspace(a, b, L + 2)[1:-1]
    elif how == "sezonier":
        y2[start:start + L] = y2[start - period:start - period + L]
    elif how == "masca_centru50":
        q = L // 4
        y2[start + q:start + L - q] = np.nan
    else:
        raise ValueError(how)
    return y2


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


# ----------------------------------------------------------------------------
# 10. Plauzibilitate: anomalia rămâne în intervalul normal al seriei?
# ----------------------------------------------------------------------------

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


# Modelele Chronos disponibile: nume scurt -> (nume în tabel, ID Hugging Face)
CHRONOS_MODELS = {
    "small": ("chronos_bolt_small", "amazon/chronos-bolt-small"),
    "base": ("chronos_bolt_base", "amazon/chronos-bolt-base"),
    "c2": ("chronos_2", "amazon/chronos-2"),
}
# Toate TSFM-urile: nume scurt -> (nume în tabel, funcție care încarcă modelul)
TSFM_MODELS = {
    **{key: (name, lambda mid=mid: make_chronos(mid))
       for key, (name, mid) in CHRONOS_MODELS.items()},
    "timesfm": ("timesfm_2p5", make_timesfm),
    "tirex": ("tirex", make_tirex),
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
    #   python echo_poc.py --main                   graficul principal din results/grid.csv
    #   python echo_poc.py --grid2 --chronos --n 100  anomalii proporționale + incertitudine
    #   python echo_poc.py --grid2 --tsfm --n 100     la fel, cu toate cele 5 TSFM-uri
    #   python echo_poc.py --plot --chronos c2 --only chronos_2 --period 48 --kind burst --len 0.5
    import sys

    if "--main" in sys.argv:
        # nu are nevoie de modele: citește doar CSV-ul produs de --grid
        import os
        import pandas as pd
        csv = arg("--csv") or "results/grid.csv"
        tabs = []
        for k in (1, 2):
            path, tab = plot_main(csv, k=k)
            tabs.append(tab)
            print(f"salvat: {path}")
        tab = pd.concat(tabs)
        os.makedirs("results", exist_ok=True)
        tab.to_csv("results/summary_ci.csv", index=False)
        print("tabel cu medii și IC: results/summary_ci.csv")
        sys.exit(0)

    if "--main2" in sys.argv:
        # graficele pentru grila proporțională (results/grid2.csv), fără GPU
        import os
        import pandas as pd
        csv = arg("--csv") or "results/grid2.csv"
        tabs = []
        for metric in ("echo", "echo_q90", "width_in", "width_abs_in"):
            for k in (1, 2):
                path, tab = plot_by_length(csv, k=k, metric=metric)
                tabs.append(tab)
                print(f"salvat: {path}")
        os.makedirs("results", exist_ok=True)
        pd.concat(tabs).to_csv("results/summary2_ci.csv", index=False)
        print("tabel cu medii și IC: results/summary2_ci.csv")
        sys.exit(0)

    if "--exceed" in sys.argv:
        # Pasul A de plauzibilitate: depășirea intervalului, fără rulări de modele
        import pandas as pd
        df = add_exceedance(pd.read_csv(arg("--csv") or "results/grid2.csv"))
        df.to_csv("results/grid2_exceed.csv", index=False)
        print("salvat: results/grid2_exceed.csv")
        print("salvat:", plot_exceed(df))
        sys.exit(0)

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
    # --tsfm = toate cele 5 modele; --tsfm timesfm,tirex = doar cele alese
    chosen = arg("--tsfm")
    if chosen is not None:
        for key in (chosen.split(",") if chosen else list(TSFM_MODELS)):
            name, loader = TSFM_MODELS[key]
            if name not in tsfm:
                tsfm[name] = loader()

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

    if "--grid2" in sys.argv:
        # Anomalii proporționale cu perioada: rafale de P/4, P/2, P și level de P.
        # Salvează și ecoul pe q0.9 + lățimea intervalului (doar la TSFM-uri).
        import os
        import pandas as pd
        n = int(arg("--n", 100))
        kinds_fracs = [("burst", 0.25), ("burst", 0.5), ("burst", 1.0), ("level", 1.0)]
        rows = []
        for P in (24, 48):
            cfg = proportional_configs(P, kinds_fracs)
            print(f"P={P}: (tip, L, offset) = {cfg}")
            rows += run({**baselines(P), **tsfm}, period=P, configs=cfg, n_series=n)
        os.makedirs("results", exist_ok=True)
        out = arg("--out") or "results/grid2.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{len(rows)} rânduri salvate în {out}\n")
        print(summarize(rows).to_string())
        sys.exit(0)

    if "--persist" in sys.argv:
        # testele de control (persistență, granițe de patch), doar pe TSFM-uri
        import os
        import pandas as pd
        n = int(arg("--n", 100))
        s_rows, curves = persistence_test(tsfm, n_series=n)
        s_rows += patch_test(tsfm, n_series=n)
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(s_rows).to_csv("results/persist_series.csv", index=False)
        pd.DataFrame(curves).to_csv("results/persist_curves.csv", index=False)
        print("salvat: results/persist_series.csv, results/persist_curves.csv")
        print("salvat:", plot_persistence(curves))
        sys.exit(0)

    if "--h3" in sys.argv:
        # H3: mascare / interpolare / imputare sezonieră a anomaliei (doar TSFM)
        import os
        import pandas as pd
        n = int(arg("--n", 100))
        df = pd.DataFrame(h3_test(tsfm, n_series=n))
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/h3.csv", index=False)
        summ = h3_summary(df)
        summ.to_csv("results/h3_summary.csv", index=False)
        print("salvat: results/h3.csv, results/h3_summary.csv")
        sys.exit(0)

    P = int(arg("--period", 24))
    forecasters = {**baselines(P), **tsfm}
    if "--plot" in sys.argv:
        # câte un grafic per model, la k=1 și k=2, apoi ne oprim
        seed = int(arg("--seed", 0))
        kind = arg("--kind", "spike")
        amp = float(arg("--amp", 5))
        amp = int(amp) if amp.is_integer() else amp   # „amp3” în numele fișierului, nu „amp3.0”
        # --len 0.5 = anomalie de P/2, cu offset proporțional ca în --grid2;
        # fără --len rămân valorile fixe de dinainte (6 pași, offset 5)
        L, off = 6, 5
        if arg("--len"):
            _, L, off = proportional_configs(P, [(kind, float(arg("--len")))])[0]
        only = arg("--only")    # ex. --only chronos_2: doar modelele alese
        for name, f in forecasters.items():
            if only and name not in only.split(","):
                continue
            for k in (1, 2):
                path, r_in, r_abs, r_out = plot_one(f, name, period=P, kind=kind,
                                                    amp=amp, k=k, seed=seed,
                                                    burst_len=L, offset=off)
                print(f"{path}: ecou={r_in:.3f}, ecou_abs={r_abs:.3f}, control={r_out:.3f}")
        sys.exit(0)
    rows = run(forecasters, period=P)
    print(summarize(rows).to_string())