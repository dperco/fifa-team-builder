from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
import logging
from app.ai_assistant.recommendation_engine import TeamRecommender 

router = APIRouter(prefix="/api/teams", tags=["teams"])
logger = logging.getLogger(__name__)

# Modelos Pydantic
class PositionCriteria(BaseModel):
    min_overall: int = Field(ge=0, le=100, default=70)
    min_pace: int | None = Field(None, ge=0, le=100)
    min_shooting: int | None = Field(None, ge=0, le=100)
    min_passing: int | None = Field(None, ge=0, le=100)
    min_defending: int | None = Field(None, ge=0, le=100)
    min_physical: int | None = Field(None, ge=0, le=100)

class PlayerResponse(BaseModel):
    id: int
    name: str
    position: str
    overall: int
    value: float
    age: int
    nationality: str
    selection_reason: str

class TeamResponse(BaseModel):
    formation: str
    description: str
    players: List[PlayerResponse]
    total_value: float
    avg_rating: float
    team_analysis: str
class TeamRequest(BaseModel):
    team_description: str = Field(..., min_length=10, max_length=500)
    team_formation: str = Field(..., pattern=r'^\d-\d(-\d)*$')
    budget: float = Field(..., gt=0)
    criteria: Dict[str, PositionCriteria]

def initialize_models():
    """Inicializa los modelos y datos necesarios"""
    try:
        from pathlib import Path
        import pandas as pd
        from sentence_transformers import SentenceTransformer
        from app.services.embeddings import generate_embeddings
        
        base_dir = Path(__file__).resolve().parent.parent.parent
        data_path = base_dir / "data" / "players_21.csv"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {data_path}")
        
        df = pd.read_csv(data_path)
        embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        embeddings_path = base_dir / "data" / "embeddings.index"
        embeddings, index = generate_embeddings(df, str(embeddings_path))
        
        return df, embedder, index
        
    except Exception as e:
        logger.error(f"Error inicializando modelos: {str(e)}")
        raise

def get_recommender():
    """Dependency que provee el recomendador configurado"""
    try:
        df, embedder, index = initialize_models()
        return TeamRecommender(df=df, embedder=embedder, index=index)
    except Exception as e:
        logger.error(f"Error inicializando recomendador: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error interno al configurar el recomendador"
        )

@router.post("/generate", response_model=TeamResponse)
async def generate_team(
    request: TeamRequest, 
    recommender: TeamRecommender = Depends(get_recommender)
):
    try:
        # Validar formación
        parts = list(map(int, request.team_formation.split('-')))
        if sum(parts) != 10:
            raise ValueError("La formación debe sumar 10 jugadores de campo")
        
        # Generar equipo
        team_data = recommender.generate_team(
            description=request.team_description,
            formation=request.team_formation,
            criteria={pos: crit.dict() for pos, crit in request.criteria.items()},
            budget=request.budget
        )
        
        # Verificar si hay resultados
        if not team_data["players"]:
            return TeamResponse(
                formation=request.team_formation,
                description=request.team_description,
                players=[],
                total_value=0,
                avg_rating=0
            )
        
        return TeamResponse(**team_data)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generando equipo: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error al generar equipo")