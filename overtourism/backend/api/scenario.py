# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from fastapi import APIRouter

from ..managers import metadata_manager, problem_manager, viewer
from ..shared.models.scenario import InputEvaluationData, OutputData, SaveData
from ..shared.utils import (
    BASE_ROUTE,
    arrange_data,
    get_id,
    get_timestamp,
    prepare_values_for_eval,
)

logger = logging.getLogger(__name__)

scenario_router = APIRouter(prefix=f"{BASE_ROUTE}/scenarios")


@scenario_router.get(
    "/{scenario_id}",
    response_model=OutputData,
    responses={
        500: {"description": "Evaluation error"},
        404: {"description": "Scenario does not exist"},
        200: {"description": "Scenario data"},
    },
)
async def get_data(
    problem_id: str,
    scenario_id: str,
    situation: str | None = None,
    session_id: str | None = None,
) -> OutputData:
    """
    Get model data.

    Parameters
    ----------
    problem_id : str
        Name of problem to evaluate.
    scenario_id : str
        Name of model to evaluate.
    situation : str
        Situation ID to evaluate.
    session_id : str
        Session id.

    Returns
    -------
    OutputData
        Model evaluation result.
    """
    try:
        scenario_manager = problem_manager.get_problem(problem_id)
        if session_id is not None:
            # Get data from reference model, do not evaluate
            old_state = scenario_manager.get_scenario_state(scenario_id)
            scenario_state = scenario_manager.evaluate_session_scenario(
                scenario_id,
                old_state.model.get_values(),
                situation=situation,
            )
            out_data = arrange_data(scenario_state.get_data(situation).data)
            values = scenario_state.model.get_values()
            return OutputData(
                problem_id=problem_id,
                scenario_id=scenario_id,
                data=out_data,
                index_diffs=scenario_state.metadata.index_diffs,
                widgets=viewer.get_widgets(values),
                editable_indexes=scenario_manager.metadata.editable_indexes,
            )

        # Get data from saved model
        out_data = arrange_data(
            scenario_manager.get_scenario_data(scenario_id, situation)
        )
        model_state = scenario_manager.get_scenario_state(scenario_id)
        values = model_state.model.get_values()
        return OutputData(
            problem_id=problem_id,
            scenario_id=scenario_id,
            data=out_data,
            index_diffs=model_state.metadata.index_diffs,
            widgets=viewer.get_widgets(values),
            editable_indexes=scenario_manager.metadata.editable_indexes,
        )
    except Exception as e:
        logger.error(
            f"Error getting data for scenario {scenario_id} in problem {problem_id}: {str(e)}"
        )
        raise e


@scenario_router.put(
    "/{scenario_id}",
    response_model=OutputData,
    responses={
        500: {"description": "Evaluation error"},
        404: {"description": "Model does not exist"},
        200: {"description": "Model data"},
    },
)
async def update_data(
    problem_id: str,
    scenario_id: str,
    data: InputEvaluationData,
    session_id: str,
) -> OutputData:
    """
    Update a model and compute new data.

    Parameters
    ----------
    problem_id : str
        Name of problem to evaluate.
    scenario_id : str
        Name of model to evaluate.
    data : InputEvaluationData
        Data to update.
    session_id : str
        Session id.

    Returns
    -------
    OutputData
        Response containing the updated model data.
    """
    try:
        scenario_manager = problem_manager.get_problem(problem_id)
        values = prepare_values_for_eval(data.values, viewer)
        scenario_state = scenario_manager.evaluate_session_scenario(
            scenario_id=scenario_id,
            values=values,
            situation=data.situation,
            ensemble_size=data.ensemble_size,
            evaluate=True,
        )
        out_data = arrange_data(scenario_state.get_data(data.situation).data)
        return OutputData(
            data=out_data,
            scenario_id=scenario_id,
            problem_id=problem_id,
            index_diffs=scenario_state.metadata.index_diffs,
            widgets=viewer.get_widgets(values),
            editable_indexes=scenario_manager.metadata.editable_indexes,
        )
    except Exception as e:
        logger.error(
            f"Error updating data for scenario {scenario_id} in problem {problem_id}: {str(e)}"
        )
        raise e


@scenario_router.post(
    "/{scenario_id}",
    response_model=dict,
    responses={
        500: {"description": "Save error"},
        200: {"description": "Model saved"},
    },
)
async def create_scenario(
    problem_id: str,
    scenario_id: str,
    session_id: str,
    data: SaveData,
    proposal_id: str | None = None,
) -> dict:
    try:
        scenario_manager = problem_manager.get_problem(problem_id)
        values = prepare_values_for_eval(data.values, viewer)
        new_id = get_id(scenario_id, session_id)
        now_timestamp = get_timestamp()
        scenario_manager.add_scenario(
            scenario_id=new_id,
            values=values,
            name=data.scenario_name,
            description=data.scenario_description,
            created=now_timestamp,
            updated=now_timestamp,
        )
        scenario_manager.evaluate_scenario(new_id)
        problem_manager.export_scenario(problem_id, new_id)
        if proposal_id is not None:
            model_state = scenario_manager.get_scenario_state(new_id)
            scenario_obj = {
                "scenario_id": new_id,
                "scenario_name": data.scenario_name,
                "scenario_description": data.scenario_description,
                "index_diffs": model_state.metadata.index_diffs,
            }
            metadata_manager.add_scenario_to_proposal(
                problem_id, proposal_id, scenario_obj
            )
        logger.info(f"Scenario created: {new_id} for problem {problem_id}")
        return {"message": "Scenario saved!"}
    except Exception as e:
        logger.error(
            f"Error creating scenario {scenario_id} for problem {problem_id}: {str(e)}"
        )
        raise e


@scenario_router.delete(
    "/{scenario_id}",
    responses={
        500: {"description": "Scenario manager error"},
        404: {"description": "Scenario does not exist"},
        200: {"description": "Scenario deleted"},
    },
)
async def delete_scenario(
    problem_id: str,
    scenario_id: str,
    proposal_id: str | None = None,
) -> None:
    """
    Delete a scenario from the manager.

    Parameters
    ----------
    problem_id : str
        ID of the problem containing the scenario.
    scenario_id : str
        ID of the scenario to delete.
    proposal_id : str, optional
        ID of the proposal to remove the scenario from, by default None.
    """
    try:
        problem_manager.delete_scenario(problem_id, scenario_id)
        if proposal_id is not None:
            metadata_manager.remove_scenario_from_proposal(
                problem_id, proposal_id, scenario_id
            )
        logger.info(f"Scenario deleted: {scenario_id} for problem {problem_id}")
    except Exception as e:
        logger.error(
            f"Error deleting scenario {scenario_id} for problem {problem_id}: {str(e)}"
        )
        raise e
