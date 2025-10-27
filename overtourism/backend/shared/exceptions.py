# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastapi import HTTPException


class OvertourismException(HTTPException):
    """Base exception for overtourism backend errors."""


class ProblemNotFound(OvertourismException):
    """Raised when a problem is not found."""

    def __init__(self, detail: str = "Problem not found") -> None:
        super().__init__(status_code=404, detail=detail)


class ScenarioDoesNotExist(OvertourismException):
    """Raised when a scenario does not exist."""

    def __init__(self, detail: str = "Scenario does not exist") -> None:
        super().__init__(status_code=404, detail=detail)


class ScenarioAlreadyExists(OvertourismException):
    """Raised when trying to create a scenario that already exists."""

    def __init__(self, detail: str = "Scenario already exists") -> None:
        super().__init__(status_code=400, detail=detail)


class FileNotFound(OvertourismException):
    """Raised when a required file is not found."""

    def __init__(self, detail: str = "File not found") -> None:
        super().__init__(status_code=404, detail=detail)


class InternalServerError(OvertourismException):
    """Raised for internal server errors."""

    def __init__(self, detail: str = "Internal server error") -> None:
        super().__init__(status_code=500, detail=detail)
