from .events import router as events_router
from .persons import router as persons_router
from .stats import router as stats_router
from .stream import router as stream_router
from .websocket import router as ws_router
from .websocket import manager as ws_manager

__all__ = [
    "events_router",
    "persons_router",
    "stats_router",
    "stream_router",
    "ws_router",
    "ws_manager",
]
