# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from abc import abstractmethod

if typing.TYPE_CHECKING:
    from ..io.classes import ProblemValues, ScenarioValues


class Store:
    """
    Abstract class to manage the export and import of data
    from problems and scenarios.
    """

    @abstractmethod
    def export_problem(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def import_problem(self, *args, **kwargs) -> ProblemValues:
        pass

    @abstractmethod
    def list_problem(self, *args, **kwargs) -> list[str]:
        pass

    @abstractmethod
    def delete_problem(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def export_scenario(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def import_scenario(self, *args, **kwargs) -> ScenarioValues:
        pass

    @abstractmethod
    def delete_scenario(self, *args, **kwargs) -> None:
        pass
