# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from fastapi import APIRouter

from ..managers import viewer
from ..shared.models.widgets import Widgets
from ..shared.utils import BASE_ROUTE

logger = logging.getLogger(__name__)

widget_router = APIRouter(prefix=BASE_ROUTE)


@widget_router.get(
    "/widgets",
    response_model=Widgets,
    responses={
        500: {"description": "View manager error"},
        200: {"description": "Widget list"},
    },
)
async def list_widgets() -> Widgets:
    """
    Get list of widgets from the view manager.

    Returns
    -------
    Widgets
        List of widgets.
    """
    try:
        return Widgets(widgets=viewer.get_widgets({}))
    except Exception as e:
        logger.error(f"Error listing widgets: {str(e)}")
        raise e
