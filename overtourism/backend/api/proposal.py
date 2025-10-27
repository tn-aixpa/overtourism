# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from fastapi import APIRouter

from ..managers import metadata_manager
from ..shared.exceptions import InternalServerError, ProblemNotFound
from ..shared.models.problem import Proposal, ProposalList
from ..shared.utils import BASE_ROUTE, get_timestamp

logger = logging.getLogger(__name__)

proposal_router = APIRouter(prefix=f"{BASE_ROUTE}/proposals")


@proposal_router.get(
    "",
    response_model=ProposalList,
    responses={
        500: {"description": "Problem manager error"},
        404: {"description": "Problem does not exist"},
        200: {"description": "Problem details"},
    },
)
async def list_proposals(problem_id: str) -> ProposalList:
    """
    Get a proposal from the manager.

    Parameters
    ----------
    problem_id : str
        ID of the problem to retrieve.
    proposal_id : str
        ID of the proposal to retrieve.

    Returns
    -------
    Proposal
        The requested proposal.
    """
    try:
        metadata = metadata_manager.list_proposal_meta(problem_id)
        p_list = []
        for proposal in metadata:
            p_list.append(Proposal(**proposal.to_dict()))
        return ProposalList(data=p_list)
    except Exception as e:
        logger.error(f"Error listing proposals for problem {problem_id}: {str(e)}")
        raise e


@proposal_router.post(
    "",
    response_model=dict,
    responses={
        500: {"description": "Problem manager error"},
        404: {"description": "Problem does not exist"},
        200: {"description": "Proposals created"},
    },
)
async def create_proposal(problem_id: str, proposal: Proposal) -> dict:
    """
    Create a new proposal for the problem.

    Parameters
    ----------
    problem_id : str
        ID of the problem to create a proposal for.
    proposal : Proposal
        Data for the new proposal.

    Returns
    -------
    dict
        A message indicating the result of the operation.
    """
    metadata = metadata_manager.read_problem_meta(problem_id)
    if metadata is None:
        raise ProblemNotFound("Problem does not exist")

    try:
        # Create proposal
        num_prop = len(metadata_manager.list_proposal_meta(problem_id))
        proposal_id = f"proposal_{num_prop + 1}"
        timestamp = get_timestamp()
        proposal_data = {
            **proposal.model_dump(),
            "proposal_id": proposal_id,
            "created": timestamp,
            "updated": timestamp,
        }
        metadata_manager.create_proposal_meta(problem_id, proposal_data)

        logger.info(f"Proposal created: {proposal_id} for problem {problem_id}")
        return {"message": "Proposal created successfully", "proposal_id": proposal_id}
    except Exception as e:
        logger.error(f"Error creating proposal for problem {problem_id}: {str(e)}")
        raise e


@proposal_router.get(
    "/{proposal_id}",
    response_model=Proposal,
    responses={
        500: {"description": "Problem manager error"},
        404: {"description": "Proposal does not exist"},
        200: {"description": "Proposal details"},
    },
)
async def read_proposal(problem_id: str, proposal_id: str) -> Proposal:
    """
    Get a proposal from the manager.

    Parameters
    ----------
    problem_id : str
        ID of the problem to retrieve.
    proposal_id : str
        ID of the proposal to retrieve.

    Returns
    -------
    Proposal
        The requested proposal.
    """
    try:
        metadata = metadata_manager.read_proposal_meta(problem_id, proposal_id)
        if metadata is None:
            raise InternalServerError("Proposal does not exist")
        return Proposal(**metadata.to_dict())
    except Exception as e:
        logger.error(
            f"Error reading proposal {proposal_id} for problem {problem_id}: {str(e)}"
        )
        raise e


@proposal_router.put(
    "/{proposal_id}",
    response_model=dict,
    responses={
        500: {"description": "Problem manager error"},
        404: {"description": "Proposal does not exist"},
        200: {"description": "Proposal updated"},
    },
)
async def update_proposal(
    problem_id: str, proposal_id: str, proposal: Proposal
) -> dict:
    """
    Update a proposal in the manager.

    Parameters
    ----------
    problem_id : str
        ID of the problem containing the proposal.
    proposal_id : str
        ID of the proposal to update.
    proposal : Proposal
        Updated data for the proposal.

    Returns
    -------
    dict
        A message indicating the result of the operation.
    """
    # Check if proposal exists
    existing_metadata = metadata_manager.read_proposal_meta(problem_id, proposal_id)
    if existing_metadata is None:
        raise InternalServerError("Proposal does not exist")

    try:
        # Update proposal
        timestamp = get_timestamp()
        proposal_data = {
            **proposal.model_dump(),
            "proposal_id": proposal_id,
            "created": existing_metadata.created,  # Keep original creation time
            "updated": timestamp,
        }
        metadata_manager.update_proposal_meta(problem_id, proposal_id, proposal_data)

        logger.info(f"Proposal updated: {proposal_id} for problem {problem_id}")
        return {"message": "Proposal updated successfully"}
    except Exception as e:
        logger.error(
            f"Error updating proposal {proposal_id} for problem {problem_id}: {str(e)}"
        )
        raise e


@proposal_router.delete(
    "/{proposal_id}",
    responses={
        500: {"description": "Problem manager error"},
        404: {"description": "Problem does not exist"},
        200: {"description": "Proposal deleted"},
    },
)
async def delete_proposal(problem_id: str, proposal_id: str) -> None:
    """
    Delete a proposal from the manager.

    Parameters
    ----------
    problem_id : str
        ID of the problem to delete a proposal from.
    proposal_id : str
        ID of the proposal to delete.
    """
    try:
        metadata_manager.delete_proposal_meta(problem_id, proposal_id)
        logger.info(f"Proposal deleted: {proposal_id} for problem {problem_id}")
    except Exception as e:
        logger.error(
            f"Error deleting proposal {proposal_id} for problem {problem_id}: {str(e)}"
        )
        raise e
