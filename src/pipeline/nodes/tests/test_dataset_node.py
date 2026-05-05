import pytest
from dora import Node

from datasets import Dataset
from pipeline.nodes.dataset_node import DatasetNode, StepStrategy


class TestDatasetNode:
    """Test dataset node."""

    @pytest.fixture
    def dataset_node(self, mocker) -> DatasetNode:
        """Mock dependencies."""
        mock_node = mocker.MagicMock(spec=Node)
        mock_data = [
            {
                "timestamp": 1000,
                "gyro_data": [1, 2, 3],
                "acc_data": [4, 5, 6],
                "imu_ts": [7, 8, 9],
                "stereo": [[10, 11, 12], [13, 14, 15]],
            }
        ]
        mock_ds = Dataset.from_list(mock_data)
        return DatasetNode(mock_ds, mock_node)

    def test_parse_step_value(self, dataset_node: DatasetNode) -> None:
        """Test that the step value is parsed correctly."""
        increment, strategy = dataset_node.parse_step_value("1")
        assert increment == 1
        assert strategy == StepStrategy.INCREMENT

        increment, strategy = dataset_node.parse_step_value("2s")
        assert increment == 2
        assert strategy == StepStrategy.SECONDS

        increment, strategy = dataset_node.parse_step_value("25%")
        assert increment == 25
        assert strategy == StepStrategy.PERCENTAGE

        increment, strategy = dataset_node.parse_step_value("dfdfsdfsfsd")
        assert increment == 0
        assert strategy == StepStrategy.NOT_DEFINED
