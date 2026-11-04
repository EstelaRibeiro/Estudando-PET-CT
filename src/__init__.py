# src/__init__.py
# Pacote principal do pipeline de análise de PET/CT.
# Importações centralizadas para facilitar o uso nos notebooks e scripts externos.

from . import io_utils
from . import visualization
from . import stats
from . import registration
from . import cardiac_views
from . import aha_bullseye

__all__ = [
    "io_utils",
    "visualization",
    "stats",
    "registration",
    "cardiac_views",
    "aha_bullseye",
]
