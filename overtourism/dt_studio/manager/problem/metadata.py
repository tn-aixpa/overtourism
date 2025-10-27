# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from overtourism.dt_studio.manager.utils.metadata import Metadata
from overtourism.dt_studio.manager.utils.utils import get_timestamp


class ProblemMetadata(Metadata):
    def __init__(
        self,
        name: str | None,
        description: str | None,
        created: str | None = None,
        updated: str | None = None,
        editable_indexes: list[str] | None = None,
    ) -> None:
        super().__init__(name, description, created, updated)
        self.editable_indexes = editable_indexes if editable_indexes is not None else []


def default_problem_metadata(
    problem_id: str,
    name: str | None = None,
    description: str | None = None,
    created: str | None = None,
    updated: str | None = None,
    editable_indexes: list[str] | None = None,
) -> ProblemMetadata:
    """
    Default problem metadata.

    Parameters
    ----------
    problem_id : str
        Unique identifier for the problem
    name : str
        Problem name
    description : str
        Problem description
    created : str
        Problem creation date
    updated : str
        Problem update date

    Returns
    -------
    ProblemMetadata
    """
    now = get_timestamp()
    name = problem_id if name is None else name
    description = f"{problem_id} problem" if description is None else description
    created = now if created is None else created
    updated = now if updated is None else updated
    editable_indexes = [] if editable_indexes is None else editable_indexes
    return ProblemMetadata(
        name=name,
        description=description,
        created=created,
        updated=updated,
        editable_indexes=editable_indexes,
    )
