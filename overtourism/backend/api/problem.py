# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import slugify
from fastapi import APIRouter

from ..managers import metadata_manager, problem_manager, viewer
from ..shared.exceptions import ProblemNotFound
from ..shared.models.problem import (
    GetProblemData,
    PostProblemData,
    ProblemList,
    UpdateProblemData,
)
from ..shared.models.scenario import ScenarioList
from ..shared.utils import BASE_ROUTE, get_timestamp, get_widget_by_group

logger = logging.getLogger(__name__)

problem_router = APIRouter(prefix=f"{BASE_ROUTE}/problems")


@problem_router.get(
    "",
    response_model=ProblemList,
    responses={
        500: {"description": "Problem manager error"},
        200: {"description": "Problem list"},
    },
)
async def list_problems() -> ProblemList:
    """
    List problems in manager.

    Returns
    -------
    ProblemList
        List of problems.
    """
    try:
        data = []
        for i in problem_manager.problems.values():
            meta = metadata_manager.read_problem_meta(i.problem_id)
            data.append(meta.to_dict())
        return ProblemList(data=data)
    except Exception as e:
        logger.error(f"Error listing problems: {str(e)}")
        raise e


@problem_router.post(
    "",
    response_model=dict,
    responses={
        500: {"description": "Problem manager error"},
        400: {"description": "Problem already exists"},
        200: {"description": "Problem created"},
    },
)
async def create_problem(data: PostProblemData) -> dict:
    """
    Create a new problem in the manager.

    Parameters
    ----------
    data : PostProblemData
        Data for the new problem.

    Returns
    -------
    dict
        Success message with problem_id.
    """
    try:
        # Generate unique problem ID
        problem_id = slugify.slugify(data.problem_name)
        timestamp = get_timestamp()

        # Get editable indexes from group
        editable_indexes = get_widget_by_group(viewer, data.groups)

        # Add problem to manager
        problem_manager.add_problem(
            problem_id=problem_id,
            name=data.problem_name,
            description=data.problem_description,
            created=timestamp,
            updated=timestamp,
            editable_indexes=editable_indexes,
        )
        problem = problem_manager.get_problem(problem_id)
        problem.add_scenario("model_0", name="Base", description="Scenario base")
        problem.evaluate_scenario("model_0")

        # Prepare metadata
        meta_dict = data.model_dump()
        meta_dict["problem_id"] = problem_id
        meta_dict["created"] = timestamp
        meta_dict["updated"] = timestamp
        meta_dict["editable_indexes"] = editable_indexes

        proposals = []
        num_proposals = 0
        for p in data.proposals:
            proposal = {
                "proposal_id": f"proposal_{num_proposals}",
                "created": timestamp,
                "updated": timestamp,
                **p,
            }
            proposals.append(proposal)
            num_proposals += 1

        meta_dict["proposals"] = proposals

        # Register metadata
        metadata_manager.create_problem_meta(problem_id, meta_dict)

        # Store problem locally
        problem_manager.export_problem(problem_id)
        metadata_manager.export_metadata(problem_id)

        logger.info(f"Problem created: {problem_id}")
        return {"message": "Problem created successfully", "problem_id": problem_id}
    except Exception as e:
        logger.error(f"Error creating problem {data.problem_name}: {str(e)}")
        raise e


@problem_router.get(
    "/{problem_id}",
    response_model=GetProblemData,
    responses={
        500: {"description": "Problem manager error"},
        404: {"description": "Problem does not exist"},
        200: {"description": "Problem details"},
    },
)
async def read_problem(problem_id: str) -> GetProblemData:
    """
    Get a problem from the manager.

    Parameters
    ----------
    problem_id : str
        ID of the problem to retrieve.

    Returns
    -------
    GetProblemData
        The requested problem.
    """
    try:
        metadata = metadata_manager.read_problem_meta(problem_id)
        if metadata is None:
            raise ProblemNotFound("Problem does not exist")
        return GetProblemData(**metadata.to_dict())
    except Exception as e:
        logger.error(f"Error reading problem {problem_id}: {str(e)}")
        raise e


@problem_router.put(
    "/{problem_id}",
    response_model=dict,
    responses={
        500: {"description": "Problem manager error"},
        404: {"description": "Problem does not exist"},
        200: {"description": "Problem updated"},
    },
)
async def update_problem(problem_id: str, data: UpdateProblemData) -> dict:
    """
    Update a problem in the manager.

    Parameters
    ----------
    problem_id : str
        ID of the problem to update.
    data : UpdateProblemData
        Updated data for the problem.

    Returns
    -------
    dict
        A message indicating the result of the operation.
    """
    try:
        # Check if problem exists
        existing_metadata = metadata_manager.read_problem_meta(problem_id)
        if existing_metadata is None:
            raise ProblemNotFound("Problem does not exist")

        # Get the problem from manager
        problem = problem_manager.get_problem(problem_id)

        # Update metadata
        timestamp = get_timestamp()
        updated_meta = existing_metadata.to_dict()
        updated_meta["updated"] = timestamp
        problem.update_metadata_field("updated", timestamp)

        # Update fields if provided
        if data.problem_name is not None:
            updated_meta["problem_name"] = data.problem_name
            problem.update_problem_name(data.problem_name)
        if data.problem_description is not None:
            updated_meta["problem_description"] = data.problem_description
            problem.update_problem_description(data.problem_description)
        if data.objective is not None:
            updated_meta["objective"] = data.objective
        if data.groups is not None:
            updated_meta["groups"] = data.groups
            # Update editable indexes based on new groups
            editable_indexes = get_widget_by_group(viewer, data.groups)
            updated_meta["editable_indexes"] = editable_indexes
            problem.update_metadata_field("editable_indexes", editable_indexes)
        if data.links is not None:
            updated_meta["links"] = data.links

        metadata_manager.update_problem_meta(problem_id, updated_meta)

        # Export updated problem
        problem_manager.export_problem(problem_id)
        metadata_manager.export_metadata(problem_id)

        logger.info(f"Problem updated: {problem_id}")
        return {"message": "Problem updated successfully"}
    except Exception as e:
        logger.error(f"Error updating problem {problem_id}: {str(e)}")
        raise e


@problem_router.delete(
    "/{problem_id}",
    responses={
        500: {"description": "Problem manager error"},
        404: {"description": "Problem does not exist"},
        200: {"description": "Problem deleted"},
    },
)
async def delete_problem(problem_id: str) -> None:
    """
    Delete a problem from the manager.

    Parameters
    ----------
    problem_id : str
        ID of the problem to delete.
    """
    try:
        # Check if problem exists
        existing_metadata = metadata_manager.read_problem_meta(problem_id)
        if existing_metadata is None:
            raise ProblemNotFound("Problem does not exist")

        problem_manager.delete_problem(problem_id)
        metadata_manager.delete_problem_meta(problem_id)
        logger.info(f"Problem deleted: {problem_id}")
    except Exception as e:
        logger.error(f"Error deleting problem {problem_id}: {str(e)}")
        raise e


@problem_router.put(
    "/refresh",
    responses={
        500: {"description": "Problem manager error"},
        200: {"description": "Problem refreshed"},
    },
)
async def refresh_problems() -> None:
    try:
        problem_manager.import_problems()
    except Exception as e:
        logger.error(f"Error refreshing problems: {str(e)}")
        raise e


@problem_router.get(
    "/{problem_id}/scenarios",
    response_model=ScenarioList,
    responses={
        500: {"description": "Problem manager error"},
        404: {"description": "Problem does not exist"},
        200: {"description": "Scenario models"},
    },
)
async def list_scenarios(problem_id: str) -> ScenarioList:
    """
    List models in manager.

    Parameters
    ----------
    problem_id : str
        Name of problem to list.

    Returns
    -------
    ModelList
        List of models.
    """
    try:
        sim = problem_manager.get_problem(problem_id)
        models = [
            {
                "problem_id": sim.problem_id,
                "scenario_id": i.scenario_id,
                "scenario_name": i.metadata.name,
                "scenario_description": i.metadata.description,
                "created": i.metadata.created,
                "updated": i.metadata.updated,
                "index_diffs": i.metadata.index_diffs,
            }
            for i in sim.instantiated_models.values()
        ]
        return ScenarioList(scenarios=models)
    except Exception as e:
        logger.error(f"Error listing scenarios for problem {problem_id}: {str(e)}")
        raise e
