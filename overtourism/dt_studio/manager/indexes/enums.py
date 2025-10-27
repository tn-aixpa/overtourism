# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum


class IndexType(Enum):
    """Enumeration of supported parameter types for model indexes.

    Attributes
    ----------
    CONSTANT : str
        Represents a fixed numeric value
    UNIFORM : str
        Represents a uniform distribution
    LOGNORM : str
        Represents a log-normal distribution
    TRIANG : str
        Represents a triangular distribution
    """

    CONSTANT = "constant"
    UNIFORM = "uniform"
    LOGNORM = "lognorm"
    TRIANG = "triang"
