"""Anneal — an agent that optimizes neural networks for the hardware they'll actually run on.

The library is organised around one idea: *nothing is estimated, everything is measured*.
An agent proposes a transform, Anneal applies it, benchmarks the result on a real
runtime, and writes the measured outcome into a ledger. The agent's next move is
conditioned on those measurements.
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
