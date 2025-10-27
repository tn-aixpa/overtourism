# SPDX-License-Identifier: Apache-2.0

from abc import ABC

from overtourism.dt_studio.manager.stores.base import Store


class TestStoreImplementation(Store):
    """Concrete implementation of Store for testing."""

    def export_problem(self, *args, **kwargs):
        return "exported_problem"

    def import_problem(self, *args, **kwargs):
        return "imported_problem"

    def list_problem(self, *args, **kwargs):
        return ["problem1", "problem2"]

    def export_scenario(self, *args, **kwargs):
        return "exported_scenario"

    def import_scenario(self, *args, **kwargs):
        return "imported_scenario"


class TestStore:
    """Test suite for Store abstract base class."""

    def test_store_can_be_instantiated(self):
        """Test that Store can be instantiated directly."""
        # Store is not actually abstract in the implementation
        store = Store()
        assert store is not None

    def test_concrete_implementation(self):
        """Test that concrete implementation works."""
        store = TestStoreImplementation()

        # Test all abstract methods can be called
        assert store.export_problem() == "exported_problem"
        assert store.import_problem() == "imported_problem"
        assert store.list_problem() == ["problem1", "problem2"]
        assert store.export_scenario() == "exported_scenario"
        assert store.import_scenario() == "imported_scenario"

    def test_abstract_methods_have_abstractmethod_decorator(self):
        """Test that methods have abstractmethod decorator."""
        # Check that methods are decorated with abstractmethod
        assert hasattr(Store.export_problem, "__isabstractmethod__")
        assert hasattr(Store.import_problem, "__isabstractmethod__")
        assert hasattr(Store.list_problem, "__isabstractmethod__")
        assert hasattr(Store.export_scenario, "__isabstractmethod__")
        assert hasattr(Store.import_scenario, "__isabstractmethod__")

    def test_inheritance(self):
        """Test that Store is a regular class (not inheriting from ABC)."""
        # Store doesn't inherit from ABC in the actual implementation
        assert not issubclass(Store, ABC)

    def test_incomplete_implementation_works(self):
        """Test that incomplete implementation can be instantiated."""

        # Since Store doesn't inherit from ABC, incomplete implementations can be instantiated
        class IncompleteStore(Store):
            def export_problem(self, *args, **kwargs):
                pass

            # Missing other required methods

        # This should work since Store is not truly abstract
        store = IncompleteStore()
        assert store is not None

    def test_method_signatures(self):
        """Test that abstract methods have correct signatures."""
        store = TestStoreImplementation()

        # Test methods accept arbitrary args and kwargs
        assert store.export_problem("arg1", kwarg1="value1") == "exported_problem"
        assert store.import_problem("arg1", "arg2") == "imported_problem"
        assert store.list_problem(filter_by="test") == ["problem1", "problem2"]
        assert store.export_scenario("arg1", "arg2", "arg3") == "exported_scenario"
        assert store.import_scenario(scenario_id="test") == "imported_scenario"
