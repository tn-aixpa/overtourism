# SPDX-License-Identifier: Apache-2.0

import pytest

from overtourism.dt_studio.manager.utils.exception import (
    ConfigurationError,
    ScenarioAlreadyExists,
    ScenarioDoesNotExist,
    ScenarioManagerException,
)


class TestExceptions:
    """Test suite for custom exception classes."""

    def test_scenario_manager_exception(self):
        """Test base ScenarioManagerException."""
        message = "Base exception message"

        with pytest.raises(ScenarioManagerException) as exc_info:
            raise ScenarioManagerException(message)

        assert str(exc_info.value) == message
        assert isinstance(exc_info.value, Exception)

    def test_scenario_already_exists(self):
        """Test ScenarioAlreadyExists exception."""
        message = "Scenario already exists"

        with pytest.raises(ScenarioAlreadyExists) as exc_info:
            raise ScenarioAlreadyExists(message)

        assert str(exc_info.value) == message
        assert isinstance(exc_info.value, ScenarioManagerException)
        assert isinstance(exc_info.value, Exception)

    def test_scenario_does_not_exist(self):
        """Test ScenarioDoesNotExist exception."""
        message = "Scenario does not exist"

        with pytest.raises(ScenarioDoesNotExist) as exc_info:
            raise ScenarioDoesNotExist(message)

        assert str(exc_info.value) == message
        assert isinstance(exc_info.value, ScenarioManagerException)
        assert isinstance(exc_info.value, Exception)

    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        message = "Configuration error occurred"

        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError(message)

        assert str(exc_info.value) == message
        assert isinstance(exc_info.value, ScenarioManagerException)
        assert isinstance(exc_info.value, Exception)

    def test_exception_inheritance_chain(self):
        """Test that all custom exceptions inherit correctly."""
        # Test that all custom exceptions inherit from ScenarioManagerException
        assert issubclass(ScenarioAlreadyExists, ScenarioManagerException)
        assert issubclass(ScenarioDoesNotExist, ScenarioManagerException)
        assert issubclass(ConfigurationError, ScenarioManagerException)

        # Test that base exception inherits from Exception
        assert issubclass(ScenarioManagerException, Exception)

    def test_exception_without_message(self):
        """Test exceptions can be raised without messages."""
        with pytest.raises(ScenarioAlreadyExists):
            raise ScenarioAlreadyExists()

        with pytest.raises(ScenarioDoesNotExist):
            raise ScenarioDoesNotExist()

        with pytest.raises(ConfigurationError):
            raise ConfigurationError()
