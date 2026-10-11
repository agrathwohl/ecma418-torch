"""Differentiable ECMA-418-2 (4th ed., 2025) roughness and fluctuation strength.

>>> from ecma418_torch import roughness, fluctuation_strength
>>> r = roughness(pressure_pa, 48000)
>>> r["single"].backward()
"""

from ecma418_torch.fluctuation import fluctuation_strength
from ecma418_torch.roughness import roughness

__all__ = ["roughness", "fluctuation_strength"]
__version__ = "0.1.0"
