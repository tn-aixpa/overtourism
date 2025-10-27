# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import glob
import typing

from overtourism.dt_studio.manager.io.utils import load_yaml, save_yaml

from .metadata import ProblemMetadata, ProposalMetadata, ScenarioMetadata

if typing.TYPE_CHECKING:
    from pathlib import Path


def delete_problem_metadata(metadata_path: Path, problem_id: str) -> None:
    """
    Delete problem metadata file.

    Parameters
    ----------
    metadata_path : Path
        Path to the metadata directory.
    problem_id : str
        ID of the problem.
    """
    (metadata_path / f"{problem_id}.yaml").unlink(missing_ok=True)


def export_problem_metadata(
    metadata_path: Path, problem_id: str, metadata: dict
) -> None:
    """
    Export problem metadata to a file.

    Parameters
    ----------
    metadata_path : Path
        Path to the metadata directory.
    problem_id : str
        ID of the problem.
    metadata : dict
        Metadata to export.
    """
    name = metadata_path / f"{problem_id}.yaml"
    save_yaml(metadata, name)


def load_problem_metadata(metadata_path: Path) -> list[dict]:
    """
    Load problem metadata from files.

    Parameters
    ----------
    metadata_path : Path
        Path to the metadata directory.

    Returns
    -------
    list[dict]
        List of problem metadata dictionaries.
    """
    out: list[dict] = []
    pattern = str(metadata_path / "*.yaml")
    for fname in glob.glob(pattern):
        try:
            dict_ = load_yaml(fname)
            out.append(dict_)
        except Exception:
            continue
    return out


def create_proposal_metadata(data: dict) -> ProposalMetadata:
    """
    Create a ProposalMetadata instance from a dictionary.

    Parameters
    ----------
    data : dict
        The input dictionary.

    Returns
    -------
    ProposalMetadata
        The created ProposalMetadata instance.
    """
    scenarios = data.get("related_scenarios", [])
    scenarios_metadata = {
        s["scenario_id"]: create_scenario_metadata(s) for s in scenarios
    }
    return ProposalMetadata(
        proposal_id=data.get("proposal_id"),
        proposal_description=data.get("proposal_description"),
        proposal_title=data.get("proposal_title"),
        created=data.get("created"),
        updated=data.get("updated"),
        resources=data.get("resources"),
        context=data.get("context"),
        impact=data.get("impact"),
        status=data.get("status"),
        related_scenarios=scenarios_metadata,
    )


def create_problem_metadata(data: dict) -> ProblemMetadata:
    """
    Create a ProblemMetadata instance from a dictionary.

    Parameters
    ----------
    data : dict
        The input dictionary.

    Returns
    -------
    ProblemMetadata
        The created ProblemMetadata instance.
    """
    return ProblemMetadata(
        problem_id=data.get("problem_id"),
        problem_name=data.get("problem_name"),
        problem_description=data.get("problem_description"),
        created=data.get("created"),
        updated=data.get("updated"),
        editable_indexes=data.get("editable_indexes"),
        groups=data.get("groups"),
        objective=data.get("objective"),
        links=data.get("links"),
        proposals=[create_proposal_metadata(p) for p in data.get("proposals", [])],
    )


def create_scenario_metadata(data: dict[str, str]) -> ScenarioMetadata:
    """
    Create a ScenarioMetadata instance from a dictionary.

    Parameters
    ----------
    data : dict[str, str]
        The input dictionary.

    Returns
    -------
    ScenarioMetadata
        The created ScenarioMetadata instance.
    """
    return ScenarioMetadata(
        scenario_id=data["scenario_id"],
        scenario_name=data.get("scenario_name"),
        scenario_description=data.get("scenario_description"),
        index_diffs=data.get("index_diffs"),
    )
