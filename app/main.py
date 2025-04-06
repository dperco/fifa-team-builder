from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app import FIFAAssistant, load_and_preprocess_data
from app.routers import teams, chat
from config import settings
import logging

from app.services.embeddings import load_embeddings_index, generate_embeddings
# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FIFA Team Builder AI Assistant",
    description="API para construir equipos de fútbol ideales con asistente inteligente",
    version="2.0.0",
    openapi_tags=[{
        "name": "Equipos",
        "description": "Endpoints para generación de equipos"
    }, {
        "name": "Chat",
        "description": "Endpoints para el asistente conversacional"
    }]
)

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(teams.router)
app.include_router(chat.router)

@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando la aplicación...")
    # Aquí podrías inicializar recursos que necesiten estar disponibles al inicio

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {
        "status": "healthy",
        "version": app.version,
        "environment": settings.ENVIRONMENT
    }