# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .data import data_router
from .problem import problem_router
from .proposal import proposal_router
from .scenario import scenario_router
from .widget import widget_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler()],
)

# Suppress watchfiles logging in development
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AIxPA Over-Tourism API",
    version="0.1.0",
    description="API for tourism indices in Trentino",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scenario_router)
app.include_router(problem_router)
app.include_router(widget_router)
app.include_router(proposal_router)
app.include_router(data_router)
