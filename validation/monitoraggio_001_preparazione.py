# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime

import slugify

from overtourism.model.setup import problem_manager
from overtourism.backend.metadata.manager import MetadataManager

PROBLEM_NAME = "validation-monitoraggio-001-problem"
PROBLEM_ID = slugify.slugify(PROBLEM_NAME)
SCENARIO_NAME = "validation-monitoraggio-001-scenario"
SCENARIO_ID = slugify.slugify(SCENARIO_NAME)
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

    # Prepare values
    values = {"excursionists_reduction_factor": 90.0}

    # Create scenario
    scenario_manager = problem_manager.get_problem(PROBLEM_ID)
    scenario_manager.add_scenario(
        scenario_id=SCENARIO_ID,
        values=values,
        name=SCENARIO_NAME,
        description="",
        created=NOW_TIMESTAMP,
        updated=NOW_TIMESTAMP,
    )

    # Export scenario
    problem_manager.export_scenario(PROBLEM_ID, SCENARIO_ID)


if __name__ == "__main__":
    main()