"""echo — primul experiment pentru tema „ecoul anomaliilor în contextul TSFM”.

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

from .data import *  # noqa: F401,F403
from .metrics import *  # noqa: F401,F403
from .baselines import *  # noqa: F401,F403
from .models import *  # noqa: F401,F403
from .grid import *  # noqa: F401,F403
from .persistence import *  # noqa: F401,F403
from .masking import *  # noqa: F401,F403
from .plausibility import *  # noqa: F401,F403
from .plots import *  # noqa: F401,F403
