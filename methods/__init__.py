# coding: utf-8
"""
Unified interface for federated-learning methods

Provide a common interface for multiple federated-learning algorithms:
- FedAvg (default)
- FedProx
- Ditto
- FedPAC
- FedRep
- FedALA
- FedAPA
- FedAS
- FedLTN
- FedSelect
"""

from .base_method import BaseFLMethod
from .registry import MethodRegistry, get_method, register_method

# Import all method implementations.
from .fedprox import FedProxMethod
from .ditto import DittoMethod
from .fedpac import FedPACMethod
from .fedrep import FedRepMethod
from .fedala import FedALAMethod
from .fedapa import FedAPAMethod
from .fedas import FedASMethod
from .fedltn import FedLTNMethod
from .fedselect import FedSelectMethod

__all__ = [
    "BaseFLMethod",
    "MethodRegistry",
    "get_method",
    "register_method",
    "FedProxMethod",
    "DittoMethod",
    "FedPACMethod",
    "FedRepMethod",
    "FedALAMethod",
    "FedAPAMethod",
    "FedASMethod",
    "FedLTNMethod",
    "FedSelectMethod",
]
