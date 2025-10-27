# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class VizIndex:
    def __init__(
        self,
        index_id: str,
        index_name: str,
        index_type: str,
        group: str,
        editable: bool = True,
        description: str | None = None,
        index_category: str | None = None,
    ) -> None:
        self.index_id = index_id
        self.index_name = index_name
        self.index_type = index_type
        self.group = group
        self.editable = editable
        self.description = description
        self.index_category = index_category

    def to_dict(self) -> dict:
        return self.__dict__

    def copy(self) -> VizIndex:
        return VizIndex(
            self.index_id,
            self.index_name,
            self.index_type,
            self.group,
            self.editable,
            self.description,
            self.index_category,
        )


class VizConstIndex(VizIndex):
    def __init__(
        self,
        index_id: str,
        index_name: str,
        index_type: str,
        group: str,
        v: float,
        min: float,
        max: float,
        step: float,
        editable: bool = True,
        description: str | None = None,
        index_category: str | None = None,
    ) -> None:
        super().__init__(
            index_id,
            index_name,
            index_type,
            group,
            editable,
            description,
            index_category,
        )
        self.v = v
        self.min = min
        self.max = max
        self.step = step

    def copy(self) -> VizConstIndex:
        return VizConstIndex(
            self.index_id,
            self.index_name,
            self.index_type,
            self.group,
            self.v,
            self.min,
            self.max,
            self.step,
            self.editable,
            self.description,
            self.index_category,
        )


class VizUniformDistIndex(VizIndex):
    def __init__(
        self,
        index_id: str,
        index_name: str,
        index_type: str,
        group: str,
        loc: float,
        scale: float,
        min: float,
        max: float,
        step: float,
        editable: bool = True,
        description: str | None = None,
        index_category: str | None = None,
    ) -> None:
        super().__init__(
            index_id,
            index_name,
            index_type,
            group,
            editable,
            description,
            index_category,
        )
        self.loc = loc
        self.scale = scale
        self.min = min
        self.max = max
        self.step = step

    def copy(self) -> VizUniformDistIndex:
        return VizUniformDistIndex(
            self.index_id,
            self.index_name,
            self.index_type,
            self.group,
            self.loc,
            self.scale,
            self.min,
            self.max,
            self.step,
            self.editable,
            self.description,
            self.index_category,
        )


class VizLognormDistIndex(VizIndex):
    def __init__(
        self,
        index_id: str,
        index_name: str,
        index_type: str,
        group: str,
        loc: float,
        scale: float,
        s: float,
        min: float,
        max: float,
        step: float,
        editable: bool = True,
        description: str | None = None,
        index_category: str | None = None,
    ) -> None:
        super().__init__(
            index_id,
            index_name,
            index_type,
            group,
            editable,
            description,
            index_category,
        )
        self.loc = loc
        self.scale = scale
        self.s = s
        self.min = min
        self.max = max
        self.step = step

    def copy(self) -> VizLognormDistIndex:
        return VizLognormDistIndex(
            self.index_id,
            self.index_name,
            self.index_type,
            self.group,
            self.loc,
            self.scale,
            self.s,
            self.min,
            self.max,
            self.step,
            self.editable,
            self.description,
            self.index_category,
        )


class VizTriangDistIndex(VizIndex):
    def __init__(
        self,
        index_id: str,
        index_name: str,
        index_type: str,
        group: str,
        loc: float,
        scale: float,
        c: float,
        min: float,
        max: float,
        step: float,
        editable: bool = True,
        description: str | None = None,
        index_category: str | None = None,
    ) -> None:
        super().__init__(
            index_id,
            index_name,
            index_type,
            group,
            editable,
            description,
            index_category,
        )
        self.loc = loc
        self.scale = scale
        self.c = c
        self.min = min
        self.max = max
        self.step = step

    def copy(self) -> VizTriangDistIndex:
        return VizTriangDistIndex(
            self.index_id,
            self.index_name,
            self.index_type,
            self.group,
            self.loc,
            self.scale,
            self.c,
            self.min,
            self.max,
            self.step,
            self.editable,
            self.description,
            self.index_category,
        )
