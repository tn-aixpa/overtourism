# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ScenarioData(BaseModel):
    problem_id: str
    scenario_id: str
    scenario_name: str
    scenario_description: str
    created: str
    updated: str
    index_diffs: dict[str, str]


class ScenarioList(BaseModel):
    scenarios: list[ScenarioData]


class InputEvaluationData(BaseModel):
    situation: str = None
    ensemble_size: int = 20
    values: dict[str, list[int | float] | int | float] = {}


class SaveData(BaseModel):
    scenario_name: str
    scenario_description: str
    values: dict[str, Any] = {}


class OutputData(BaseModel):
    problem_id: str
    scenario_id: str
    data: dict[str, Any] = {}
    index_diffs: dict[str, str]
    widgets: dict = None
    editable_indexes: list[str] = None
