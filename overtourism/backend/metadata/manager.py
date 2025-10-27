# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from pathlib import Path

from .utils import (
    create_problem_metadata,
    create_proposal_metadata,
    create_scenario_metadata,
    delete_problem_metadata,
    export_problem_metadata,
    load_problem_metadata,
)

if typing.TYPE_CHECKING:
    from .metadata import ProblemMetadata, ProposalMetadata


class MetadataManager:
    """
    Manager for problem and proposal metadata stored on disk.

    Parameters
    ----------
    metadata_path : str
        Path to the metadata directory or file used by the manager.

    Attributes
    ----------
    metadata : dict[str, ProblemMetadata]
        In-memory cache of loaded problem metadata keyed by problem id.
    metadata_path : pathlib.Path
        Resolved path to the metadata storage location.
    """

    def __init__(self, metadata_path: str):
        self.metadata: dict[str, ProblemMetadata] = {}
        self.metadata_path = Path(metadata_path)
        self._load_metadata()

    def _load_metadata(self) -> None:
        """
        Load metadata from disk into the in-memory cache.

        Reads all problem metadata from `self.metadata_path` using
        :func:`load_problem_metadata` and populates ``self.metadata`` with
        :class:`ProblemMetadata` instances keyed by their ``id``.
        """
        self.metadata = {
            md["problem_id"]: create_problem_metadata(md)
            for md in load_problem_metadata(self.metadata_path)
        }

    def export_metadata(self, problem_id: str) -> None:
        """
        Export a single problem's metadata to disk.

        Parameters
        ----------
        problem_id : str
            Identifier of the problem to export.
        """
        export_problem_metadata(
            self.metadata_path, problem_id, self.metadata[problem_id].to_dict()
        )

    def remove_metadata(self, problem_id: str) -> None:
        """
        Remove a problem's metadata from the disk.

        Parameters
        ----------
        problem_id : str
            Identifier of the problem to remove.
        """
        delete_problem_metadata(self.metadata_path, problem_id)

    def refresh_metadata(self) -> None:
        """
        Remove all metadata from the in-memory cache and disk.
        """
        self.metadata.clear()
        self._load_metadata()

    ################################
    # Problem metadata
    ################################

    def create_problem_meta(self, problem_id: str, value: dict) -> None:
        """
        Create and set problem metadata in the in-memory cache.

        Parameters
        ----------
        problem_id : str
            Identifier under which the metadata will be stored.
        value : dict
            Keyword-compatible dictionary used to construct a
            :class:`ProblemMetadata` via :func:`create_problem_metadata`.
        """
        self.metadata[problem_id] = create_problem_metadata(value)
        self.export_metadata(problem_id)

    def read_problem_meta(self, problem_id: str) -> ProblemMetadata | None:
        """
        Retrieve a problem's metadata from the cache.

        Parameters
        ----------
        problem_id : str
            Problem identifier to look up.

        Returns
        -------
        ProblemMetadata or None
            The cached :class:`ProblemMetadata` if present, otherwise ``None``.
        """
        return self.metadata.get(problem_id)

    def update_problem_meta(self, problem_id: str, value: dict) -> None:
        """
        Update a problem's metadata in the in-memory cache.

        Parameters
        ----------
        problem_id : str
            Identifier of the problem to update.
        value : dict
            Keyword-compatible dictionary used to construct a
            :class:`ProblemMetadata` via :func:`create_problem_metadata`.
        """
        self.metadata[problem_id] = create_problem_metadata(value)
        self.export_metadata(problem_id)

    def delete_problem_meta(self, problem_id: str) -> None:
        """
        Delete a problem's metadata from the cache.

        Parameters
        ----------
        problem_id : str
            Problem identifier to delete.
        """
        self.metadata.pop(problem_id, None)
        delete_problem_metadata(self.metadata_path, problem_id)

    def list_problem_meta(self) -> dict[str, ProblemMetadata]:
        """
        Get the entire in-memory metadata cache.

        Returns
        -------
        dict[str, ProblemMetadata]
            Mapping of problem id to :class:`ProblemMetadata`.
        """
        return self.metadata

    ################################
    # Proposal metadata
    ################################

    def create_proposal_meta(self, problem_id: str, value: dict) -> None:
        """
        Add a proposal metadata entry to a problem's proposals list.

        If the target problem does not exist in the cache this is a no-op.

        Parameters
        ----------
        problem_id : str
            Problem identifier where the proposal will be added.
        value : dict
            Keyword-compatible dictionary used to construct a
            :class:`ProposalMetadata` via :func:`create_proposal_metadata`.
        """
        problem_metadata = self.read_problem_meta(problem_id)
        if problem_metadata:
            problem_metadata.proposals.append(create_proposal_metadata(value))
            self.export_metadata(problem_id)

    def read_proposal_meta(
        self, problem_id: str, proposal_id: str
    ) -> ProposalMetadata | None:
        """
        Retrieve a proposal's metadata for a given problem.

        Parameters
        ----------
        problem_id : str
            Problem identifier containing the proposal.
        proposal_id : str
            Proposal identifier to search for.

        Returns
        -------
        ProposalMetadata or None
            The matching proposal metadata, or ``None`` if not found.
        """
        problem_metadata = self.read_problem_meta(problem_id)
        if problem_metadata:
            return next(
                (p for p in problem_metadata.proposals if p.proposal_id == proposal_id),
                None,
            )
        return None

    def delete_proposal_meta(self, problem_id: str, proposal_id: str) -> None:
        """
        Delete a proposal's metadata for a given problem.

        Parameters
        ----------
        problem_id : str
            Problem identifier containing the proposal.
        proposal_id : str
            Proposal identifier to delete.
        """
        problem_metadata = self.read_problem_meta(problem_id)
        if problem_metadata:
            problem_metadata.proposals = [
                p for p in problem_metadata.proposals if p.proposal_id != proposal_id
            ]
            self.export_metadata(problem_id)

    def list_proposal_meta(self, problem_id: str) -> list[ProposalMetadata]:
        """
        Get all proposals metadata for a given problem.

        Parameters
        ----------
        problem_id : str
            Problem identifier to look up.

        Returns
        -------
        list[ProposalMetadata]
            List of proposal metadata for the specified problem.
        """
        problem_metadata = self.read_problem_meta(problem_id)
        if problem_metadata:
            return problem_metadata.proposals
        return []

    def update_proposal_meta(
        self, problem_id: str, proposal_id: str, value: dict
    ) -> None:
        """
        Update a proposal's metadata for a given problem.

        Parameters
        ----------
        problem_id : str
            Problem identifier containing the proposal.
        proposal_id : str
            Proposal identifier to update.
        value : dict
            Keyword-compatible dictionary used to construct a
            :class:`ProposalMetadata` via :func:`create_proposal_metadata`.
        """
        problem_metadata = self.read_problem_meta(problem_id)
        if problem_metadata:
            for i, proposal in enumerate(problem_metadata.proposals):
                if proposal.proposal_id == proposal_id:
                    problem_metadata.proposals[i] = create_proposal_metadata(value)
                    self.export_metadata(problem_id)
                    break

    def add_scenario_to_proposal(
        self,
        problem_id: str,
        proposal_id: str,
        scenario_info: dict,
    ) -> None:
        """
        Add a scenario identifier to a proposal's scenarios list.

        Parameters
        ----------
        problem_id : str
            Problem identifier containing the proposal.
        proposal_id : str
            Proposal identifier to update.
        scenario_id : str
            Scenario identifier to add to the proposal.
        """
        problem_metadata = self.read_problem_meta(problem_id)
        if not problem_metadata:
            return

        scenario_id = scenario_info["scenario_id"]

        for proposal in problem_metadata.proposals:
            if proposal.proposal_id == proposal_id:
                if scenario_id in proposal.related_scenarios:
                    # Scenario already exists, update metadata
                    meta = proposal.related_scenarios[scenario_id]
                    meta.scenario_name = scenario_info.get(
                        "scenario_name", meta.scenario_name
                    )
                    meta.scenario_description = scenario_info.get(
                        "scenario_description", meta.scenario_description
                    )
                else:
                    meta = create_scenario_metadata(scenario_info)
                    proposal.related_scenarios[scenario_id] = meta

                self.export_metadata(problem_id)
                break

    def remove_scenario_from_proposal(
        self,
        problem_id: str,
        proposal_id: str,
        scenario_id: str,
    ) -> None:
        """
        Remove a scenario identifier from a proposal's scenarios list.

        Parameters
        ----------
        problem_id : str
            Problem identifier containing the proposal.
        proposal_id : str
            Proposal identifier to update.
        scenario_id : str
            Scenario identifier to remove from the proposal.
        """
        problem_metadata = self.read_problem_meta(problem_id)
        if not problem_metadata:
            return

        for proposal in problem_metadata.proposals:
            if proposal.proposal_id == proposal_id:
                proposal.related_scenarios.pop(scenario_id, None)
                self.export_metadata(problem_id)
                break
