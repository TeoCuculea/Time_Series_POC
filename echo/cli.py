"""Linia de comandă: python echo_poc.py --grid2 --tsfm ... (vezi README.md)."""

import numpy as np

from .data import proportional_configs
from .baselines import batchify, context_parrot, seasonal_mean, seasonal_median, seasonal_naive
from .models import CHRONOS_MODELS, TSFM_MODELS, make_chronos
from .grid import add_exceedance, run, summarize
from .persistence import patch_test, persistence_test
from .masking import h3_summary, h3_test, mask_cost_test
from .plausibility import plausibility_summary, plausibility_test
from .plots import plot_by_length, plot_exceed, plot_main, plot_one, plot_persistence, plot_plausibility


def arg(name, default=None):
    """Valoarea de după `name` în linia de comandă (ex. --seed 3), altfel default.

    Dacă opțiunea există dar nu are valoare (ex. `--chronos` singur), întoarce ""."""
    import sys
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    nxt = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
    return "" if nxt.startswith("--") else nxt


def main():
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
    # --tsfm = cele 5 modele de bază; --tsfm t5 sau --tsfm moirai = doar cele alese
    # (moirai doar cu interpretorul din .venv-moirai)
    chosen = arg("--tsfm")
    if chosen is not None:
        default = ["small", "base", "c2", "timesfm", "tirex"]
        for key in (chosen.split(",") if chosen else default):
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

    if "--plaus" in sys.argv:
        # Pasul B de plauzibilitate: poziție în ciclu × în/peste interval × direcție
        import os
        import pandas as pd
        df = pd.DataFrame(plausibility_test(tsfm, n_series=int(arg("--n", 100))))
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/plausibility.csv", index=False)
        summ, excl = plausibility_summary(df)
        summ.to_csv("results/plausibility_summary.csv", index=False)
        excl.to_csv("results/plausibility_excluded.csv", index=False)
        print("salvat: results/plausibility*.csv")
        print("salvat:", *plot_plausibility(df))
        sys.exit(0)

    if "--maskcost" in sys.argv:
        # costul unei alarme false: mască NaN pe un segment normal
        import os
        import pandas as pd
        df = pd.DataFrame(mask_cost_test(tsfm, n_series=int(arg("--n", 100))))
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/mask_cost.csv", index=False)
        print("salvat: results/mask_cost.csv")
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
