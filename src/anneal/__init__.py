"""Anneal — measured INT8 quantization for the hardware a model will actually run on.

The library is organised around one idea: *nothing is estimated, everything is measured*.
It diagnoses why a model breaks under INT8 on a given target, applies exact fixes
(equalisation, surrogates, safeguards), and verifies each candidate recipe by benchmarking
it on a real runtime and recording the measured outcome in a ledger.
"""

__version__ = "0.1.0"

from anneal.core.artifact import ModelArtifact
from anneal.core.ledger import Ledger, Trial
from anneal.core.measure import Measurement, Benchmarker
from anneal.core.targets import Target, get_target, list_targets

__all__ = [
    "ModelArtifact",
    "Ledger",
    "Trial",
    "Measurement",
    "Benchmarker",
    "Target",
    "get_target",
    "list_targets",
    "__version__",
]
