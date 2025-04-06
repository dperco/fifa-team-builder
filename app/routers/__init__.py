"""
Módulo de routers para la API del Asistente de Fútbol FIFA

Exporta todos los routers disponibles para ser incluidos en la aplicación principal
"""

from fastapi import APIRouter
from .teams import router as teams_router
from .chat import router as chat_router

# Router principal que agrupa todos los routers
main_router = APIRouter()

# Incluye todos los routers específicos
main_router.include_router(
    teams_router,
    prefix="/api/teams",
    tags=["Equipos"]
)

main_router.include_router(
    chat_router,
    prefix="/api/chat",
    tags=["Chat"]
)

# Exporta el router principal para su uso en main.py
__all__ = ["main_router"]