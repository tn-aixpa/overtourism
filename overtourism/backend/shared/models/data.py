# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel


class OutputGeoJSON(BaseModel):
    data: dict


class InputGeoJSONOptions(BaseModel):
    date: str
    hours: str
    day_type: str
    user: str
