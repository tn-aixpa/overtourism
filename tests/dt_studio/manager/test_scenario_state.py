# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from overtourism.dt_studio.manager.scenario.state import ScenarioState, SituationData


class TestSituationData:
    """Test suite for SituationData class."""

    def test_init_with_data(self):
        """Test SituationData initialization with data."""
        mock_data = Mock()
        situation_data = SituationData(mock_data)

        assert situation_data.data == mock_data

    def test_init_without_data(self):
        """Test SituationData initialization without data."""
        situation_data = SituationData()

        assert situation_data.data is None


class TestScenarioState:
    """Test suite for ScenarioState class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.scenario_id = "test_scenario"
        self.mock_metadata = Mock()

        self.scenario_state = ScenarioState(
            model=self.mock_model,
            scenario_id=self.scenario_id,
            metadata=self.mock_metadata,
        )

        yield

    def test_init(self):
        """Test ScenarioState initialization."""
        assert self.scenario_state.scenario_id == self.scenario_id
        assert self.scenario_state.model == self.mock_model
        assert self.scenario_state.metadata == self.mock_metadata
        assert isinstance(self.scenario_state.situations, dict)
        assert len(self.scenario_state.situations) == 0
        assert self.scenario_state.is_evaluating is False

    def test_add_data_with_data(self):
        """Test adding situation data with model output."""
        situation_id = "test_situation"
        mock_data = Mock()

        self.scenario_state.add_data(situation_id, mock_data)

        assert situation_id in self.scenario_state.situations
        situation_data = self.scenario_state.situations[situation_id]
        assert isinstance(situation_data, SituationData)
        assert situation_data.data == mock_data

    def test_add_data_without_data(self):
        """Test adding situation data without model output."""
        situation_id = "test_situation"

        self.scenario_state.add_data(situation_id)

        assert situation_id in self.scenario_state.situations
        situation_data = self.scenario_state.situations[situation_id]
        assert isinstance(situation_data, SituationData)
        assert situation_data.data is None

    def test_get_data_existing(self):
        """Test getting existing situation data."""
        situation_id = "test_situation"
        mock_data = Mock()
        self.scenario_state.add_data(situation_id, mock_data)

        result = self.scenario_state.get_data(situation_id)

        assert isinstance(result, SituationData)
        assert result.data == mock_data

    def test_get_data_nonexistent(self):
        """Test getting non-existent situation data creates empty entry."""
        situation_id = "nonexistent_situation"

        result = self.scenario_state.get_data(situation_id)

        assert isinstance(result, SituationData)
        assert result.data is None
        assert situation_id in self.scenario_state.situations

    def test_get_all_data(self):
        """Test getting all situation data."""
        situation1 = "situation1"
        situation2 = "situation2"
        mock_data1 = Mock()
        mock_data2 = Mock()

        self.scenario_state.add_data(situation1, mock_data1)
        self.scenario_state.add_data(situation2, mock_data2)

        all_data = self.scenario_state.get_all_data()

        assert isinstance(all_data, dict)
        assert len(all_data) == 2
        assert situation1 in all_data
        assert situation2 in all_data
        assert all_data[situation1].data == mock_data1
        assert all_data[situation2].data == mock_data2

    def test_copy(self):
        """Test copying scenario state."""
        # Add some data to original
        situation_id = "test_situation"
        mock_data = Mock()
        self.scenario_state.add_data(situation_id, mock_data)
        self.scenario_state.is_evaluating = True

        copied_state = self.scenario_state.copy()

        # Check that copy has same attributes
        assert copied_state.scenario_id == self.scenario_state.scenario_id
        assert copied_state.model == self.scenario_state.model
        assert copied_state.metadata == self.scenario_state.metadata

        # Check that situations are copied
        assert len(copied_state.situations) == len(self.scenario_state.situations)
        assert situation_id in copied_state.situations

        # Check that is_evaluating is not copied (it's a new state)
        assert copied_state.is_evaluating is False

        # Verify it's a different object
        assert copied_state is not self.scenario_state
        assert copied_state.situations is not self.scenario_state.situations

    def test_copy_preserves_situation_references(self):
        """Test that copy preserves situation data references."""
        situation_id = "test_situation"
        mock_data = Mock()
        self.scenario_state.add_data(situation_id, mock_data)

        copied_state = self.scenario_state.copy()

        # The situation data should be the same reference
        original_situation = self.scenario_state.situations[situation_id]
        copied_situation = copied_state.situations[situation_id]
        assert original_situation is copied_situation

    def test_multiple_situations(self):
        """Test managing multiple situations."""
        situations = ["sit1", "sit2", "sit3"]
        mock_data = [Mock(), Mock(), Mock()]

        # Add multiple situations
        for i, sit_id in enumerate(situations):
            self.scenario_state.add_data(sit_id, mock_data[i])

        # Verify all are present
        all_data = self.scenario_state.get_all_data()
        assert len(all_data) == 3

        for i, sit_id in enumerate(situations):
            assert sit_id in all_data
            assert all_data[sit_id].data == mock_data[i]

    def test_overwrite_situation_data(self):
        """Test overwriting existing situation data."""
        situation_id = "test_situation"
        original_data = Mock()
        new_data = Mock()

        # Add original data
        self.scenario_state.add_data(situation_id, original_data)
        assert self.scenario_state.get_data(situation_id).data == original_data

        # Overwrite with new data
        self.scenario_state.add_data(situation_id, new_data)
        assert self.scenario_state.get_data(situation_id).data == new_data

        # Should still have only one entry
        assert len(self.scenario_state.situations) == 1
