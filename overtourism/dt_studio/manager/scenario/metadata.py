# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from overtourism.dt_studio.manager.utils.metadata import Metadata
from overtourism.dt_studio.manager.utils.utils import get_timestamp


class ScenarioMetadata(Metadata):
    def __init__(
        self,
        name: str | None,
        description: str | None,
        created: str | None = None,
        updated: str | None = None,
        index_diffs: dict[str, str] | None = None,
    ) -> None:
        super().__init__(name, description, created, updated)
        self.index_diffs = index_diffs if index_diffs is not None else {}


def default_scenario_metadata(
    scenario_id: str,
    name: str | None = None,
    description: str | None = None,
    created: str | None = None,
    updated: str | None = None,
    index_diffs: dict[str, str] | None = None,
) -> ScenarioMetadata:
    """
    Default scenario metadata.

    Parameters
    ----------
    scenario_id : str
        Unique identifier for the scenario
    name : str
        Scenario name
    description : str
        Scenario description
    created : str
        Scenario creation date
    updated : str
        Scenario update date
    index_diffs : dict
            Index diffs

    Returns
    -------
    ScenarioMetadata
    """
    now = get_timestamp()
    name = scenario_id if name is None else name
    description = f"{scenario_id} scenario" if description is None else description
    created = now if created is None else created
    updated = now if updated is None else updated
    index_diffs = {} if index_diffs is None else index_diffs
    return ScenarioMetadata(
        name=name,
        description=description,
        created=created,
        updated=updated,
        index_diffs=index_diffs,
    )
