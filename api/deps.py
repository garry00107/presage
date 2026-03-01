"""
FastAPI dependency injection for shared resources.
All routes access the SessionFactory and stores through these deps.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException

from core.session.factory import SessionFactory
from core.session.manager import SessionManager

# Module-level singleton — set during app startup
_factory: SessionFactory | None = None


def set_factory(factory: SessionFactory) -> None:
    global _factory
    _factory = factory


def get_factory() -> SessionFactory:
    if _factory is None:
        raise RuntimeError("SessionFactory not initialized. Call set_factory() at startup.")
    return _factory


async def get_session(session_id: str) -> SessionManager:
    factory = get_factory()
    session = await factory.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session


FactoryDep = Annotated[SessionFactory, Depends(get_factory)]

