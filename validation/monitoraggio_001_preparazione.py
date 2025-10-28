# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime

import slugify

from overtourism.model.setup import problem_manager

PROBLEM_NAME = "validation-problem"
PROBLEM_ID = slugify.slugify(PROBLEM_NAME)
SCENARIO_NAME = "validation-scenario"
SCENARIO_ID = slugify.slugify(SCENARIO_NAME)
NOW_TIMESTAMP = datetime.now().astimezone().isoformat()


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