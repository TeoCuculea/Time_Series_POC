"""
echo_poc.py — punctul de intrare pentru experimentele „ecoul anomaliilor în TSFM”.

Codul propriu-zis e în pachetul echo/ (structura e descrisă în README.md):
  echo/data.py          serii sintetice, anomalii, reparații (NaN, interpolare...)
  echo/metrics.py       ecou, lățimea intervalului, depășire, bootstrap
  echo/baselines.py     seasonal naive / mean / median, context parroting
  echo/models.py        Chronos-Bolt, Chronos-2, Chronos T5, TimesFM, TiRex, Moirai
  echo/grid.py          grilele --grid și --grid2
  echo/persistence.py   persistență vs. copiere de fază, granițe de patch
  echo/masking.py       H3 (mascare) și costul unei alarme false
  echo/plausibility.py  anomalia în / peste intervalul seriei
  echo/plots.py         toate graficele
  echo/cli.py           opțiunile din linia de comandă

Exemple:
  python echo_poc.py --grid2 --tsfm --n 100
  python echo_poc.py --main2
"""

from echo import *  # noqa: F401,F403  (ca `from echo_poc import make_series` să meargă și acum)
from echo.cli import main

if __name__ == "__main__":
    main()
