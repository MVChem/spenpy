"""Core SPEN encoding matrix computations."""

from spenpy._legacy.core.sinc import math_sinc
from spenpy._legacy.core.matrix import calcSRMatrixApprox, calcInvA

__all__ = ["math_sinc", "calcSRMatrixApprox", "calcInvA"]
