# Note de laborator: modele, versiuni, licențe

## Modele folosite

| Model | ID Hugging Face | Pachet (versiune) | Licență |
|---|---|---|---|
| Chronos-Bolt small | `amazon/chronos-bolt-small` | chronos-forecasting 2.3.2 | Apache-2.0 |
| Chronos-Bolt base | `amazon/chronos-bolt-base` | chronos-forecasting 2.3.2 | Apache-2.0 |
| Chronos-2 | `amazon/chronos-2` | chronos-forecasting 2.3.2 | Apache-2.0 |
| TimesFM 2.5 (200M) | `google/timesfm-2.5-200m-pytorch` | timesfm 3.0.2 | Apache-2.0 |
| TiRex | `NX-AI/TiRex` | tirex-ts 1.4.2 | **NXAI Community License** |
| Chronos T5 small | `amazon/chronos-t5-small` | chronos-forecasting 2.3.2 | Apache-2.0 (model card) |
| Moirai 2.0 small | `Salesforce/moirai-2.0-R-small` | uni2ts 2.0.0 (cod Apache-2.0) | **CC BY-NC 4.0** (ponderile, model card) |

Licențele Apache-2.0 sunt cele declarate de pachete / model card-uri;
de verificat încă o dată pe model card înainte de citarea finală.

## Moirai 2.0: licență și mediu separat

- Ponderile sunt sub **CC BY-NC 4.0**: uz necomercial permis (o disertație e
  în regulă), cu atribuire. Codul uni2ts e Apache-2.0.
- uni2ts 2.0.0 cere `numpy~=1.26`, `scipy~=1.11`, `torch<2.5`: fără pachete
  pentru Python 3.13, iar torch < 2.5 nu suportă GPU-ul Blackwell (sm_120).
  Soluția: venv separat `.venv-moirai` (Python 3.13, torch 2.11+cu128),
  uni2ts instalat cu `--no-deps`, dependențele în versiuni recente.
  **Nu e combinația testată de autori**: rezultatele Moirai trebuie verificate
  (eroare de prognoză la nivelul zgomotului, cuantile ordonate) înainte de grilă.
- Rulare: `.venv-moirai\Scripts\python.exe echo_poc.py --grid2 --tsfm moirai ...`
  (are nevoie și de pandas / matplotlib în acel venv).

## Chronos T5: eșantionare

- Prognoza vine din `num_samples` traiectorii aleatoare; seed-ul e resetat la
  fiecare apel, deci geamănul curat și cel cu anomalie primesc aceeași
  secvență aleatoare (zgomotul se anulează parțial în diferență).
- 20 de eșantioane (implicit): ecoul mediu variază cu ±0,002 între seed-uri;
  pe o singură serie, abaterea între seed-uri e ~0,03.
- Memorie: serii × eșantioane generate deodată ≤ 1500 (altfel OOM pe 16 GB).

## TiRex: NXAI Community License

TiRex **nu** e sub o licență open-source standard. Licența (NXAI Community
License Agreement, 27 mai 2025, NXAI GmbH, Linz) e derivată din Meta Llama 3
Community License. Pe scurt:

- utilizarea, inclusiv pentru cercetare, e permisă gratuit (§1.a);
- dacă **redistribui** materialele NXAI (cod, ponderi) sau un produs care le
  folosește: copie a licenței + mențiunea „Built with technology from NXAI”
  + un fișier Notice cu textul de atribuire (§1.b);
- o licență comercială separată e necesară doar pentru firme cu venit anual
  peste 100 mil. € care integrează TiRex într-un produs comercial (§2).

Acest repo **nu** redistribuie cod sau ponderi TiRex: pachetul se instalează
cu pip, iar ponderile se descarcă de pe Hugging Face. În disertație: citez
modelul și menționez licența (NXAI Community License), nu doar „open-source”.
Textul complet: `.venv/Lib/site-packages/tirex_ts-1.4.2.dist-info/licenses/LICENSE`.

## Setări de rulare care pot conta

- Toate modelele în float32 (metrica e o diferență între prognoze aproape identice).
- Chronos-2: `cross_learning=False`, ca seriile din lot să nu-și împrumute informație.
- TimesFM: `torch_compile=False` (lipsește Triton pe Windows); `ForecastConfig`
  cu setările recomandate în README (normalize_inputs, continuous quantile head,
  flip invariance, fix_quantile_crossing). TimesFM înlocuiește intern valorile
  NaN prin interpolare liniară.
- TiRex: backend `torch` (kernelurile CUDA dedicate cer pachetul `xlstm`).
- Bolt, Chronos-2, TimesFM, TiRex sunt deterministe la inferență: rerulările
  dau exact aceleași cifre (verificat pentru Chronos, diferență 0).

## Taguri

- `grid2-2026-11-22`: grila proporțională, doar modelele Chronos.
- `grid2-5models-2026-11-22`: grila proporțională cu toate cele 5 TSFM-uri,
  testele de persistență / patch și lățimea absolută a intervalului.
