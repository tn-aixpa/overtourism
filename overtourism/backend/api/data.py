# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from fastapi import APIRouter

from ..managers import overtourism_indexes_data_loader
from ..shared.utils import BASE_ROUTE

logger = logging.getLogger(__name__)

data_router = APIRouter(prefix=f"{BASE_ROUTE}/data")


@data_router.get(
    "/overtourism/indexes/categories",
    response_model=dict,
    responses={
        500: {"description": "Data loader error"},
        200: {"description": "Overtourism categories list"},
    },
)
async def get_overtourism_categories_list() -> dict:
    """
    Get list of overtourism categories.

    Returns
    -------
    dict
        Overtourism categories list.
    """
    try:
        return overtourism_indexes_data_loader.get_categories()
    except Exception as e:
        logger.error(f"Error getting overtourism categories list: {str(e)}")
        raise e


@data_router.get(
    "/overtourism/indexes/list",
    response_model=dict,
    responses={
        500: {"description": "Data loader error"},
        200: {"description": "Overtourism indexes list"},
    },
)
async def get_overtourism_indexes_list(category: str = "") -> dict:
    """
    Get list of overtourism indexes (of a given category, if specified).

    Returns
    -------
    dict
        Overtourism indexes list.
    """
    try:
        return overtourism_indexes_data_loader.get_list(category)
    except Exception as e:
        logger.error(f"Error getting overtourism indexes list: {str(e)}")
        raise e


@data_router.get(
    "/overtourism/indexes/data",
    response_model=dict,
    responses={
        500: {"description": "Data loader error"},
        404: {"description": "Data not found"},
        200: {"description": "Overtourism index data"},
    },
)
async def get_overtourism_indexes_data(dataframe: str) -> dict:
    """
    Get overtourism data for a given index.

    Returns
    -------
    dict
        Overtourism index data.
    """
    try:
        return overtourism_indexes_data_loader.get_dataframe(dataframe)
    except Exception as e:
        logger.error(f"Error getting overtourism indexes data: {str(e)}")
        raise e


@data_router.get(
    "/overtourism/indexes/map",
    response_model=dict,
    responses={
        500: {"description": "Data loader error"},
        404: {"description": "Data not found"},
        200: {"description": "Overtourism index map"},
    },
)
async def get_overtourism_indexes_map(map: str) -> dict:
    """
    Get overtourism one of the indexes' maps.

    Returns
    -------
    dict
        Overtourism index map.
    """
    try:
        return overtourism_indexes_data_loader.get_map(map)
    except Exception as e:
        logger.error(f"Error getting overtourism indexes map: {str(e)}")
        raise e
