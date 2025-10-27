# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .utils import Dictable


class Metadata(Dictable):
    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        created: str | None = None,
        updated: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.created = created
        self.updated = updated
