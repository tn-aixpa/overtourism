# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class Metadata:
    """
    Base class for all metadata classes.
    """

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if value is None or key.startswith("_"):
                continue
            if isinstance(value, list):
                filtered_list = [
                    v.to_dict() if hasattr(v, "to_dict") else v
                    for v in value
                    if v is not None
                ]
                if filtered_list:
                    result[key] = filtered_list
            elif hasattr(value, "to_dict"):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


class ProblemMetadata(Metadata):
    """
    Class to hold metadata about a problems.
    """

    def __init__(
        self,
        problem_id: str,
        problem_name: str,
        problem_description: str,
        created: str | None = None,
        updated: str | None = None,
        editable_indexes: list[str] | None = None,
        groups: list[str] | None = None,
        objective: str | None = None,
        links: list[str] | None = None,
        proposals: list[ProposalMetadata] | None = None,
    ) -> None:
        self.problem_id: str = problem_id
        self.problem_name: str = problem_name
        self.problem_description: str = problem_description
        self.created: str = created
        self.updated: str = updated
        self.editable_indexes: list[str] = editable_indexes
        self.groups: list[str] = groups
        self.objective: str = objective
        self.links: list[str] = links
        self.proposals: list[ProposalMetadata] = proposals


class ProposalMetadata(Metadata):
    """
    Class to hold metadata about a proposals.
    """

    def __init__(
        self,
        proposal_id: str,
        proposal_description: str | None = None,
        proposal_title: str | None = None,
        created: str | None = None,
        updated: str | None = None,
        resources: list[str] | None = None,
        context: str | None = None,
        impact: str | None = None,
        status: str | None = None,
        related_scenarios: dict[str, ScenarioMetadata] | None = None,
    ):
        self.proposal_id: str = proposal_id
        self.proposal_description: str = proposal_description
        self.proposal_title: str = proposal_title
        self.created: str = created
        self.updated: str = updated
        self.resources: list[str] = resources
        self.context: str = context
        self.impact: str = impact
        self.status: str = status
        self.related_scenarios: dict[str, ScenarioMetadata] = related_scenarios

    def to_dict(self) -> dict:
        result = super().to_dict()
        if "related_scenarios" in result:
            result["related_scenarios"] = [
                v.to_dict() for v in self.related_scenarios.values()
            ]
        return result


class ScenarioMetadata(Metadata):
    """
    Class to hold metadata about a scenario.
    """

    def __init__(
        self,
        scenario_id: str,
        scenario_name: str | None = None,
        scenario_description: str | None = None,
        index_diffs: dict | None = None,
    ) -> None:
        self.scenario_id: str = scenario_id
        self.scenario_name: str = scenario_name
        self.scenario_description: str = scenario_description
        self.index_diffs: dict = index_diffs
