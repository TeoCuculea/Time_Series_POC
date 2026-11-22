"""Generarea seriilor sintetice și a anomaliilor."""

import numpy as np


def make_series(n, period, noise=0.1, seed=0, shift=0):
    """Serie sezonieră: sinus + o armonică + zgomot gaussian. Amplitudine ~1.

    shift (implicit 0 = exact ca înainte): deplasează TOATĂ unda cu `shift`
    pași, fără să-i schimbe forma și fără să schimbe zgomotul. Îl folosim ca
    să punem anomalia pe vale / pantă / vârf fără să-i mutăm poziția în timp."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) + shift
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
