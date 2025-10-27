# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum


class StoreType(Enum):
    """Enumeration of supported I/O store types."""

    LOCAL = "local"
