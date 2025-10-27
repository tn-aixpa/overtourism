# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class ScenarioManagerException(Exception):
    pass


class ScenarioAlreadyExists(ScenarioManagerException):
    pass


class ScenarioDoesNotExist(ScenarioManagerException):
    pass


class ConfigurationError(ScenarioManagerException):
    pass
