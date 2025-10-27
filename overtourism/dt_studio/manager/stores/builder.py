# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing

from .enums import StoreType
from .local import LocalIOStore

if typing.TYPE_CHECKING:
    from .base import Store


class StoreBuilder:
    """
    Factory builder with register pattern for creating IO stores.
    """

    _stores: dict[str, Store] = {}

    @classmethod
    def register(cls, store_type: str, store_class: Store) -> None:
        """
        Register a new store type.

        Parameters
        ----------
        store_type : str
            Type of IO store to register
        store_class : Store
            Class to instantiate for this store type
        """
        cls._stores[store_type] = store_class

    @classmethod
    def create(cls, store_type: str, *args, **kwargs) -> Store:
        """
        Create an IO store instance based on the specified type.

        Parameters
        ----------
        store_type : str
            Type of IO store to create

        Returns
        -------
        LocalIOStore
            Instance of the appropriate IO store

        Raises
        ------
        ValueError
            If an unregistered store type is specified
        """
        try:
            return cls._stores[store_type](*args, **kwargs)
        except KeyError:
            raise ValueError(f"Unregistered IO store type: {store_type}")


# Register the default store
io_builder = StoreBuilder()
io_builder.register(StoreType.LOCAL.value, LocalIOStore)
