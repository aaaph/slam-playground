import pytest

from pipeline.nodes.dataset_node import DatasetNode, StepStrategy


class TestDatasetNode:
    """Test dataset node."""

    @pytest.fixture
    def dataset_node(self) -> DatasetNode:
        """Create a DatasetNode instance without initializing dataset iteration."""
        return DatasetNode.__new__(DatasetNode)

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
