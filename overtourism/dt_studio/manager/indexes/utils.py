# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing

from civic_digital_twins.dt_model import (
    ConstIndex,
    LognormDistIndex,
    TriangDistIndex,
    UniformDistIndex,
)

if typing.TYPE_CHECKING:
    from civic_digital_twins.dt_model import InstantiatedModel
    from civic_digital_twins.dt_model.symbols.index import Index


def get_diff_str(idx: Index, value: dict | float) -> str | None:
    def tostr(v):
        if v == int(v):
            return str(int(v))
        return str(v)

    if isinstance(idx, ConstIndex):
        if idx.v == value:
            return None
        return f"{tostr(idx.v)} -> {tostr(value)}"
    if isinstance(idx, (UniformDistIndex, TriangDistIndex)):
        if idx.loc == value["loc"] and idx.scale == value["scale"]:
            return None
        return f"{tostr(idx.loc)}-{tostr(idx.loc + idx.scale)} -> {tostr(value['loc'])}-{tostr(value['loc'] + value['scale'])}"
    if isinstance(idx, LognormDistIndex):
        if idx.loc == value["loc"] and idx.scale == value["scale"]:
            return None
        return f"~{tostr(idx.scale)} -> ~{tostr(value['scale'])}"
    return None


def get_diff(model: InstantiatedModel) -> dict[str, str]:
    changes = {}
    for k, v in model.get_values().items():
        if isinstance(v, (float, int)):
            changes[k] = v
        else:
            changes[k] = v.kwds

    res = {}
    for idx in model.abs.indexes + model.abs.capacities:
        if idx.name in changes:
            diff_str = get_diff_str(idx, changes[idx.name])
            if diff_str is not None:
                res[idx.name] = diff_str
    return res
