# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime


def get_timestamp() -> str:
    """
    Get the current timestamp timezoned.

    Returns
    -------
    str
        The current timestamp.
    """
    return datetime.now().astimezone().isoformat()


class Dictable:
    """Base class providing dictionary conversion functionality.

    This class implements a method to convert nested objects into dictionaries,
    handling lists and nested Dictable objects recursively.

    Methods
    -------
    to_dict() -> dict
        Convert the object and its nested attributes to a dictionary
    """

    def to_dict(self) -> dict:
        """Convert object to dictionary recursively.

        Returns
        -------
        dict
            Dictionary representation of the object
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                result[key] = [
                    item.to_dict() if hasattr(item, "to_dict") else item
                    for item in value
                ]
            elif hasattr(value, "to_dict"):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
