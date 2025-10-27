# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class ProposalStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class RelatedScenario(BaseModel):
    scenario_id: str
    scenario_name: str | None = None
    scenario_description: str | None = None
    index_diffs: dict | None = None


class Proposal(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    proposal_id: str | None = None
    proposal_description: str | None = None
    proposal_title: str | None = None
    resources: list[str] = []
    created: str | None = None
    updated: str | None = None
    context: str | None = None
    impact: str | None = None
    status: ProposalStatus = ProposalStatus.DRAFT
    related_scenarios: list[RelatedScenario] = []


class PostProblemData(BaseModel):
    problem_name: str
    problem_description: str
    created: str | None = None
    updated: str | None = None
    objective: str | None = None
    groups: list[str] = []
    links: list[str] = []
    proposals: list[Proposal] = []


class UpdateProblemData(BaseModel):
    problem_id: str | None = None
    problem_name: str | None = None
    problem_description: str | None = None
    created: str | None = None
    updated: str | None = None
    objective: str | None = None
    groups: list[str] = None
    links: list[str] = None
    proposals: list[Proposal] = None
    editable_indexes: list[str] = None


class GetProblemData(PostProblemData):
    problem_id: str
    editable_indexes: list[str] = []
    groups: list[str] = []


class ProblemList(BaseModel):
    data: list[GetProblemData]


class ProposalList(BaseModel):
    data: list[Proposal]
