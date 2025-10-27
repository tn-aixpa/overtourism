# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from overtourism.dt_studio.viewer.models import ViewType
from overtourism.dt_studio.viewer.viewer import ModelViewer


class TestModelViewer:
    """Test suite for ModelViewer class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.test_config = {
            "indexes": [
                {
                    "index_id": "test_index",
                    "group": "test_group",
                    "index_type": "constant",
                }
            ]
        }

        with (
            patch("overtourism.dt_studio.viewer.viewer.load_yaml") as mock_load_yaml,
            patch(
                "overtourism.dt_studio.viewer.viewer.build_indexes_from_config"
            ) as mock_build_indexes,
        ):
            mock_load_yaml.return_value = self.test_config
            mock_build_indexes.return_value = [Mock()]

            self.viewer = ModelViewer("/test/path")

        yield

    @patch("overtourism.dt_studio.viewer.viewer.load_yaml")
    @patch("overtourism.dt_studio.viewer.viewer.build_indexes_from_config")
    def test_init(self, mock_build_indexes, mock_load_yaml):
        """Test ModelViewer initialization."""
        mock_load_yaml.return_value = self.test_config
        mock_indexes = [Mock()]
        mock_build_indexes.return_value = mock_indexes

        viewer = ModelViewer("/test/path")

        mock_load_yaml.assert_called_once_with("/test/path")
        mock_build_indexes.assert_called_once_with(self.test_config)
        assert viewer.indexes == mock_indexes

    @patch("overtourism.dt_studio.viewer.viewer.bidimensional_figure")
    def test_viz_bidimensional(self, mock_bidimensional_figure):
        """Test visualization with bidimensional view."""
        mock_data = Mock()
        mock_data.sample_x = [1, 2, 3]
        mock_data.sample_y = [4, 5, 6]
        mock_data.constraint_curves = {}
        mock_data.x_max = 10
        mock_data.y_max = 10

        mock_view = Mock()
        mock_view.view_type = ViewType.BIDIMENSIONAL.value
        mock_view.constraint = None

        # Mock uncertainty data
        with patch.object(
            self.viewer, "_get_uncertainty", return_value=[0.1, 0.2, 0.3]
        ):
            mock_figure = Mock()
            mock_bidimensional_figure.return_value = mock_figure

            result = self.viewer.viz(mock_data, mock_view, "Test Title")

            assert result == mock_figure
            mock_bidimensional_figure.assert_called_once()

    @patch("overtourism.dt_studio.viewer.viewer.unidimensional_figure")
    def test_viz_unidimensional(self, mock_unidimensional_figure):
        """Test visualization with unidimensional view."""
        mock_data = Mock()
        mock_data.sample_x = [1, 2, 3]
        mock_data.sample_y = [4, 5, 6]
        mock_data.usage = [0.5, 0.6, 0.7]
        mock_data.usage_uncertainty = [0.05, 0.06, 0.07]
        mock_data.capacity_mean = 1.0

        mock_view = Mock()
        mock_view.view_type = ViewType.UNIDIMENSIONAL.value
        mock_view.constraint = None

        with (
            patch.object(
                self.viewer, "_get_1d_capacity_capacity_mean", return_value=1.0
            ),
            patch.object(
                self.viewer,
                "_get_1d_sample",
                return_value=(
                    [1, 2, 3],
                    [4, 5, 6],
                    [0.5, 0.6, 0.7],
                    [0.05, 0.06, 0.07],
                ),
            ),
            patch.object(self.viewer, "_get_1d_ymax_top", return_value=(10, 15)),
        ):
            mock_figure = Mock()
            mock_unidimensional_figure.return_value = mock_figure

            result = self.viewer.viz(mock_data, mock_view, "Test Title")

            assert result == mock_figure
            mock_unidimensional_figure.assert_called_once()

    def test_viz_unknown_view_type(self):
        """Test visualization with unknown view type raises error."""
        mock_data = Mock()
        mock_view = Mock()
        mock_view.view_type = "UNKNOWN"

        with pytest.raises(ValueError) as exc_info:
            self.viewer.viz(mock_data, mock_view, "Test Title")

        assert "Unknown view type" in str(exc_info.value)

    def test_get_uncertainty_no_constraint(self):
        """Test getting uncertainty data without constraint."""
        mock_data = Mock()
        mock_data.uncertainty = [0.1, 0.2, 0.3]

        result = self.viewer._get_uncertainty(mock_data)

        assert result == [0.1, 0.2, 0.3]

    def test_get_uncertainty_with_constraint(self):
        """Test getting uncertainty data with constraint."""
        mock_data = Mock()
        mock_data.uncertainty_by_constraint = {"test_constraint": [0.4, 0.5, 0.6]}

        result = self.viewer._get_uncertainty(mock_data, "test_constraint")

        assert result == [0.4, 0.5, 0.6]

    def test_get_1d_sample(self):
        """Test getting 1D sample data."""
        mock_data = Mock()
        mock_data.sample_x = [3, 1, 2]
        mock_data.sample_y = [6, 4, 5]
        mock_data.usage = [0.9, 0.3, 0.6]
        mock_data.usage_uncertainty = [0.09, 0.03, 0.06]

        sample_t, sample_e, usage, usage_uncertainty = self.viewer._get_1d_sample(
            mock_data
        )

        # Data should be sorted by usage
        sample_t = list(sample_t)
        sample_e = list(sample_e)
        usage = list(usage)
        usage_uncertainty = list(usage_uncertainty)

        assert usage == [0.3, 0.6, 0.9]  # Sorted by usage
        assert sample_t == [1, 2, 3]  # Corresponding x values
        assert sample_e == [4, 5, 6]  # Corresponding y values
        assert usage_uncertainty == [0.03, 0.06, 0.09]

    def test_get_1d_sample_with_constraint(self):
        """Test getting 1D sample data with constraint."""
        mock_data = Mock()
        mock_data.sample_x = [3, 1, 2]
        mock_data.sample_y = [6, 4, 5]
        mock_data.usage_by_constraint = {"test_constraint": [0.9, 0.3, 0.6]}
        mock_data.usage_uncertainty_by_constraint = {
            "test_constraint": [0.09, 0.03, 0.06]
        }

        sample_t, sample_e, usage, usage_uncertainty = self.viewer._get_1d_sample(
            mock_data, "test_constraint"
        )

        usage = list(usage)
        assert usage == [0.3, 0.6, 0.9]  # Sorted by usage

    def test_get_1d_capacity_capacity_mean_no_constraint(self):
        """Test getting capacity mean without constraint."""
        mock_data = Mock()
        mock_data.capacity_mean = 2.5

        result = self.viewer._get_1d_capacity_capacity_mean(mock_data)

        assert result == 2.5

    def test_get_1d_capacity_capacity_mean_with_constraint(self):
        """Test getting capacity mean with constraint."""
        mock_data = Mock()
        mock_data.capacity_mean_by_constraint = {"test_constraint": 3.7}

        result = self.viewer._get_1d_capacity_capacity_mean(
            mock_data, "test_constraint"
        )

        assert result == 3.7

    def test_get_1d_ymax_top(self):
        """Test calculating ymax and top values."""
        usage = [0.5, 0.6, 0.7]
        capacity_mean = 1.0
        sample_t = [1, 2, 3]
        sample_e = [0.1, 0.2, 0.3]

        ymax, top = self.viewer._get_1d_ymax_top(
            usage, capacity_mean, sample_t, sample_e
        )

        # ymax should be 1.2 * max(max(usage), capacity_mean)
        expected_ymax = int(max(max(usage), capacity_mean) * 1.2)
        assert ymax == expected_ymax

        # top calculation is more complex, just verify it's an integer
        assert isinstance(top, int)

    def test_get_groups(self):
        """Test getting groups from indexes."""
        mock_index1 = Mock()
        mock_index1.group = "group1"
        mock_index2 = Mock()
        mock_index2.group = "group2"
        mock_index3 = Mock()
        mock_index3.group = "group1"

        self.viewer.indexes = [mock_index1, mock_index2, mock_index3]
        # Rebuild groups after changing indexes
        self.viewer.groups = self.viewer._build_groups()

        groups = self.viewer.get_groups()

        # Should have 2 groups
        assert len(groups) == 2

        # Check group structure
        group_ids = [g["id"] for g in groups]
        assert "group1" in group_ids
        assert "group2" in group_ids

    def test_get_groups_with_none_group(self):
        """Test getting groups when some indexes have None group."""
        mock_index1 = Mock()
        mock_index1.group = None
        mock_index2 = Mock()
        mock_index2.group = "group1"

        self.viewer.indexes = [mock_index1, mock_index2]
        # Rebuild groups after changing indexes
        self.viewer.groups = self.viewer._build_groups()

        groups = self.viewer.get_groups()

        # Should have 2 groups: "general" and "group1"
        assert len(groups) == 2
        group_ids = [g["id"] for g in groups]
        assert "general" in group_ids
        assert "group1" in group_ids

    def test_get_widgets(self):
        """Test getting widgets configuration."""
        mock_index1 = Mock()
        mock_index1.group = "group1"
        mock_index1.index_id = "index1"
        mock_index1.index_type = "constant"
        mock_index1.copy.return_value = mock_index1
        mock_index1.to_dict.return_value = {"id": "index1", "type": "constant"}

        mock_index2 = Mock()
        mock_index2.group = "group1"
        mock_index2.index_id = "index2"
        mock_index2.index_type = "distribution"
        mock_index2.copy.return_value = mock_index2
        mock_index2.to_dict.return_value = {"id": "index2", "type": "distribution"}

        self.viewer.indexes = [mock_index1, mock_index2]

        vals = {"index1": 5.0, "index2": Mock(kwds={"loc": 1.0, "scale": 0.5})}

        widgets = self.viewer.get_widgets(vals)

        assert "group1" in widgets
        assert len(widgets["group1"]) == 2

    def test_build_groups(self):
        """Test building groups from indexes."""
        mock_index1 = Mock()
        mock_index1.group = "group1"
        mock_index2 = Mock()
        mock_index2.group = None

        self.viewer.indexes = [mock_index1, mock_index2]

        groups = self.viewer._build_groups()

        # Should create groups and add parameters
        assert isinstance(groups, list)
        assert len(groups) > 0

    def test_get_group(self):
        """Test getting group structure."""
        group_id = "test_group"

        group = self.viewer._get_group(group_id)

        expected = {
            "id": group_id,
            "label": group_id,
            "parameters": [],
        }
        assert group == expected

    def test_get_point_data(self):
        """Test getting point data for visualization."""
        uncertainty_matrix = [
            {"tourists": 100, "excursionists": 50, "index": 0.5},
            {"tourists": 200, "excursionists": 100, "index": 0.7},
            {"tourists": 150, "excursionists": 75, "index": 0.6},
        ]

        result = self.viewer._get_point_data(uncertainty_matrix)

        expected = {
            "x": [100, 200, 150],
            "y": [50, 100, 75],
            "z": [0.5, 0.7, 0.6],
        }
        assert result == expected
