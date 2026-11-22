# Ecoul anomaliilor în TSFM — POC

**Întrebarea:** dacă în istoricul unei serii sezoniere a existat o anomalie,
o repetă un Time Series Foundation Model în prognoză, la aceeași fază
(„ecoul”)?

**Metoda (serii gemene):** generăm o serie curată și o copie identică cu o
anomalie injectată cu *k* perioade înainte de momentul prognozei. Prognozăm
ambele cu același model. Diferența dintre prognoze, împărțită la mărimea
anomaliei, e `d(h)`. **Ecoul** = media lui `d` în fereastra unde anomalia
s-ar repeta (1 = copiere completă, ca seasonal naive; 0 = nicio urmă).

## Structura codului

| fișier | ce conține |
|---|---|
| `echo_poc.py` | punctul de intrare; toate comenzile de mai jos |
| `echo/data.py` | `make_series` (serie sintetică), `inject` (anomalia), `layout` (unde cade anomalia și fereastra de ecou), `proportional_configs`, `repair` (NaN / interpolare / imputare sezonieră) |
| `echo/metrics.py` | `echo_ratio` (ecou, ecou_abs, control), `width_change` (lățimea intervalului q0.1–q0.9), `exceedance`, `bootstrap_ci`, `split_q` |
| `echo/baselines.py` | seasonal naive / mean / median, context parroting |
| `echo/models.py` | wrapper-ele TSFM; toate întorc `forecast(batch_ctx, h) -> (lot, orizont, 3)` |
| `echo/grid.py` | `run` (grila de bază), `summarize`, `add_exceedance` |
| `echo/persistence.py` | persistență vs. copiere de fază, granițe de patch |
| `echo/masking.py` | H3 (mascarea anomaliei), costul unei alarme false |
| `echo/plausibility.py` | anomalia în / peste intervalul seriei, doză–răspuns |
| `echo/plots.py` | toate graficele |
| `echo/cli.py` | opțiunile din linia de comandă |
| `NOTES.md` | versiuni, licențe, setări de rulare care pot conta |

Importurile vechi merg în continuare: `from echo_poc import make_series`.

## Comenzi

Modelele se aleg cu `--chronos` (cele 3 Chronos) sau `--tsfm` (cele 5 de bază:
Bolt small, Bolt base, Chronos-2, TimesFM 2.5, TiRex). Comenzile fără model
lucrează doar pe CSV-uri și nu au nevoie de GPU.

| comandă | ce face | produce |
|---|---|---|
| `--plot --chronos small --kind burst --len 0.5 --period 48` | o serie: prognoze, interval, `d(h)` | `figures/one_*.png` |
| `--grid --chronos` | grila inițială: rafală 6 pași, P=24/48, 30 serii | `results/grid.csv` |
| `--main` | grafic ecou cu IC bootstrap (din `grid.csv`) | `figures/main_*`, `results/summary_ci.csv` |
| `--grid2 --tsfm --n 100` | anomalii proporționale cu P (P/4, P/2, P, level) + incertitudine | `results/grid2.csv` |
| `--main2` | grafice pe lungimea anomaliei (din `grid2.csv`) | `figures/bylen_*`, `results/summary2_ci.csv` |
| `--persist --tsfm` | persistență vs. fază; offset ±3 (patch) | `results/persist_*.csv`, `figures/persist_dh.png` |
| `--h3 --tsfm` | H3: ecoul după mască / interpolare / imputare | `results/h3*.csv` |
| `--maskcost --tsfm` | costul unei alarme false (3 umpleri ale golului) | `results/mask_cost.csv` |
| `--exceed` | plauzibilitate A: depășirea intervalului (din `grid2.csv`) | `results/grid2_exceed.csv`, `figures/exceed_echo.png` |
| `--plaus --tsfm` | plauzibilitate B: 8 poziții × în/peste interval × drop | `results/plausibility*.csv`, `figures/plaus_*` |

## Jurnal — 22 noiembrie 2026

Fiecare pas: întrebarea → ce am rulat → ce a ieșit. Cifrele complete sunt în
CSV-urile din `results/`; tag-urile marchează versiunea codului care le-a produs.

1. **Mediul + baseline-uri** (`d9b458d`). Seasonal naive dă 1, media pe faze
   0,125, mediana ≈ 0: metrica funcționează.
2. **Chronos-Bolt small, base, Chronos-2** (`f6cb9de`, `6487bdf`). Rafală de
   6 pași: ecou ~0,1 sau mai mic; la vârfuri (1 punct) și la k=2 ≈ 0.
3. **Grila inițială + grafic cu IC** (`1dcfe7f`, `3229f8c`). Niciun model nu
   atinge 0,3. Problemă de design: rafala de 6 pași e P/4 la P=24 dar P/8 la P=48.
4. **Anomalii proporționale cu perioada** (`6206acb`, tag `grid2-2026-11-22`).
   **Durata contează mai mult decât amplitudinea:** de la P/2 în sus, Bolt are
   ecou 0,3–0,6; anomaliile scurte sunt filtrate. Intervalul q0.1–q0.9 se
   lărgește mult după anomalie.
5. **TimesFM 2.5 + TiRex, teste de control** (`77f455d`, tag
   `grid2-5models-2026-11-22`).
   - Persistență: efectul apare în fereastra de fază, nu la începutul
     orizontului → **copiere de fază**, cu greutate mai mare pentru anomaliile
     recente (Bolt small: 1,00 → 0,92 → 0,59 → 0,31 cu distanța).
   - Level de o perioadă: toate modelele urmăresc nivelul (ecou ≈ 1) — nu e memorie de fază.
   - Modelele noi (Chronos-2 la P=48, TimesFM, TiRex) au ecou mic.
   - Granițele de patch nu explică bimodalitatea Chronos-2.
6. **H3 — mascarea anomaliei** (`63c65e5`). La modelele cu ecou (Bolt,
   Chronos-2 la P=24) masca NaN reduce ecoul cu 97–99%, iar o mască parțială
   (50% central) tot cu 63–88%. NaN e mai bun decât interpolarea liniară.
7. **Plauzibilitate** (`8d7e9fe`, `0d2bbcc`, tag `plausibility-2026-11-22`).
   Ipoteza: modelele copiază anomaliile care rămân în intervalul normal al
   seriei. Regula de decizie e îndeplinită doar la TiRex, P=48, pe pantă.
8. **Doză–răspuns pe amplitudine absolută + costul golului** (`e50b7a8`).
   - Curbele pozițiilor se suprapun (ΔR² poziție 0,004–0,07): **ecoul depinde
     de mărimea anomaliei, nu de faptul că iese din interval.**
   - Alarmă falsă (mască pe un segment normal): Chronos și TiRex pierd puțin
     (+1–11% MAE); TimesFM +127–131%, pentru că umple golul cu o linie dreaptă.
     Umplut cu y[t−P], costul scade la ~1%.
9. **Chronos T5 + Moirai 2.0 — în lucru** (`791f373`). Wrapper-e scrise, încă
   nerulate pe grilă. Moirai rulează din `.venv-moirai` (vezi `NOTES.md`).
   Chronos T5 cu 20 de eșantioane: ecoul mediu stabil între seed-uri
   (0,242–0,244 la P=24, P/2, 5σ); testul cu 100 de eșantioane n-a terminat.
10. **Refactorizare** — `echo_poc.py` (1.500 de linii) împărțit în pachetul
    `echo/`. Verificat: grid, main, main2, exceed, H3, plauzibilitate,
    persistență și `plot_one` dau exact aceleași cifre ca înainte.

## Ipotezele, unde suntem

- **H1** (ecou ≥ 0,3 la ≥ 2 din 5 TSFM): îndeplinită de o singură familie
  (Chronos-Bolt), la rafale de P/2 sau mai lungi.
- **H2** (ecoul diferă pe familii): da, calitativ — Bolt copiază faza,
  Chronos-2 / TimesFM / TiRex sunt mult mai robuste.
- **H3** (mascarea reduce ecoul ≥ 50%): da, acolo unde există ecou.
