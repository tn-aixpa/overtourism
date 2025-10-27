# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from overtourism.dt_studio.manager.indexes.enums import IndexType
from overtourism.dt_studio.viewer.indexes.index import (
    VizConstIndex,
    VizIndex,
    VizLognormDistIndex,
    VizTriangDistIndex,
    VizUniformDistIndex,
)


def build_indexes_from_config(config: dict) -> list[VizIndex]:
    """
    Build a list of VizIndex instances from a configuration dictionary.

    Args:
        config: Dictionary containing index configurations

    Returns:
        List of VizIndex instances
    """
    indexes = []

    for idx_config in config["indexes"]:
        try:
            index_type = idx_config["index_type"]
            common_args = {
                "index_id": idx_config["index_id"],
                "index_name": idx_config["index_name"],
                "index_type": index_type,
                "group": idx_config["group"],
                "editable": idx_config.get("editable", True),
                "description": idx_config.get("description"),
                "index_category": idx_config.get("index_category"),
            }

            if index_type == IndexType.CONSTANT.value:
                indexes.append(
                    VizConstIndex(
                        **common_args,
                        v=idx_config["v"],
                        min=idx_config["min"],
                        max=idx_config["max"],
                        step=idx_config["step"],
                    )
                )
            elif index_type == IndexType.UNIFORM.value:
                indexes.append(
                    VizUniformDistIndex(
                        **common_args,
                        loc=idx_config["loc"],
                        scale=idx_config["scale"],
                        min=idx_config["min"],
                        max=idx_config["max"],
                        step=idx_config["step"],
                    )
                )
            elif index_type == IndexType.LOGNORM.value:
                indexes.append(
                    VizLognormDistIndex(
                        **common_args,
                        loc=idx_config["loc"],
                        scale=idx_config["scale"],
                        s=idx_config["s"],
                        min=idx_config["min"],
                        max=idx_config["max"],
                        step=idx_config["step"],
                    )
                )
            elif index_type == IndexType.TRIANG.value:
                indexes.append(
                    VizTriangDistIndex(
                        **common_args,
                        loc=idx_config["loc"],
                        scale=idx_config["scale"],
                        c=idx_config["c"],
                        min=idx_config["min"],
                        max=idx_config["max"],
                        step=idx_config["step"],
                    )
                )
            else:
                raise ValueError(f"Unknown index type: {index_type}")

        except KeyError as e:
            raise ValueError(f"Missing key in index configuration: {e}")
    return indexes
