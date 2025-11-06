# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime

import slugify

from scipy import stats

import pandas as pd

from overtourism.model.setup import problem_manager
from overtourism.backend.metadata.manager import MetadataManager

PROBLEM_NAME = "validation-overtourism-002"
PROBLEM_ID = slugify.slugify(PROBLEM_NAME)
SCENARIO_0_NAME = "validation-overtourism-002-scenario-base"
SCENARIO_0_ID = slugify.slugify(SCENARIO_0_NAME)
SCENARIO_0_DESCRIPTION = "Scenario base"
SCENARIO_1_NAME = "validation-overtourism-002-scenario-1"
SCENARIO_1_ID = slugify.slugify(SCENARIO_1_NAME)
SCENARIO_1_DESCRIPTION = "Aumento posti auto"
SCENARIO_2_NAME = "validation-overtourism-002-scenario-2"
SCENARIO_2_ID = slugify.slugify(SCENARIO_2_NAME)
SCENARIO_2_DESCRIPTION = "Riduzione flussi escursionisti"
NOW_TIMESTAMP = datetime.now().astimezone().isoformat()
METADATA_PATH = problem_manager.store.folder.parent / "metadata"
METADATA_PATH.mkdir(parents=True, exist_ok=True)

def main():

    # Add problem to manager
    problem_manager.add_problem(
        problem_id=PROBLEM_ID,
        name=PROBLEM_NAME,
        description="",
        created=NOW_TIMESTAMP,
        updated=NOW_TIMESTAMP,
        editable_indexes=[],
    )

    # Export problem
    problem_manager.export_problem(PROBLEM_ID)

    # Create and export metadata
    metadata_manager = MetadataManager(METADATA_PATH)
    meta_dict = {
        "problem_id": PROBLEM_ID,
        "problem_name": PROBLEM_NAME,
        "problem_description": "",
        "created": NOW_TIMESTAMP,
        "updated": NOW_TIMESTAMP,
    }
    metadata_manager.create_problem_meta(PROBLEM_ID, meta_dict)
    metadata_manager.export_metadata(PROBLEM_ID)

    # Prepare to create scenarios
    scenario_manager = problem_manager.get_problem(PROBLEM_ID)

    # Create scenario 1
    scenario_manager.add_scenario(
        scenario_id=SCENARIO_0_ID,
        values={},
        name=SCENARIO_0_NAME,
        description=SCENARIO_0_DESCRIPTION,
        created=NOW_TIMESTAMP,
        updated=NOW_TIMESTAMP,
    )

    # Export scenario 1
    problem_manager.export_scenario(PROBLEM_ID, SCENARIO_0_ID)

    # Prepare values for scenario 1
    values_1 = {"available_parking_spaces": stats.uniform(loc=550, scale=200)}

    # Create scenario 1
    scenario_manager.add_scenario(
        scenario_id=SCENARIO_1_ID,
        values=values_1,
        name=SCENARIO_1_NAME,
        description=SCENARIO_1_DESCRIPTION,
        created=NOW_TIMESTAMP,
        updated=NOW_TIMESTAMP,
    )

    # Export scenario 1
    problem_manager.export_scenario(PROBLEM_ID, SCENARIO_1_ID)

    # Prepare values for scenario 2
    values_2 = {"excursionists_reduction_factor": 80.0}

    # Create scenario 2
    scenario_manager.add_scenario(
        scenario_id=SCENARIO_2_ID,
        values=values_2,
        name=SCENARIO_2_NAME,
        description=SCENARIO_2_DESCRIPTION,
        created=NOW_TIMESTAMP,
        updated=NOW_TIMESTAMP,
    )

    # Export scenario 2
    problem_manager.export_scenario(PROBLEM_ID, SCENARIO_2_ID)

    # Execute the two scenarios
    scenario_manager.evaluate_scenario(SCENARIO_0_ID)
    data_0 = scenario_manager.get_scenario_data(SCENARIO_0_ID)
    scenario_manager.evaluate_scenario(SCENARIO_1_ID)
    data_1 = scenario_manager.get_scenario_data(SCENARIO_1_ID)
    scenario_manager.evaluate_scenario(SCENARIO_2_ID)
    data_2 = scenario_manager.get_scenario_data(SCENARIO_2_ID)

    index_map = {
        'overtourism_level': 'Indice complessivo (% giorni di overtourism)',
        'constraint level parcheggi': 'Indice di utilizzo parcheggi (% giorni di superamento capacità)',
        'constraint level spiaggia': 'Indice di utilizzo spiaggia (% giorni di superamento capacità)',
        'constraint level alberghi': 'Indice di utilizzo albergio (% giorni di superamento capacità)',
        'constraint level ristoranti': 'Indice di utilizzo ristoranti (% giorni di superamento capacità)',
    }

    df = pd.DataFrame(data=[
        [data_0.kpis[i]['level'], data_1.kpis[i]['level'], data_2.kpis[i]['level']] for i in index_map],
        index=list(index_map.values()),
        columns=[SCENARIO_0_DESCRIPTION, SCENARIO_1_DESCRIPTION, SCENARIO_2_DESCRIPTION],
    )
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None,
                           'max_colwidth', None):
        print (df)


if __name__ == "__main__":
    main()