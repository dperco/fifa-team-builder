from pydantic import BaseSettings
from typing import List

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: List[str] = ["http://localhost:5173"]
    DATA_PATH: str = "data/players_21.csv"
    EMBEDDINGS_PATH: str = "models/embeddings.faiss"
    MODEL_NAME: str = "paraphrase-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"

settings = Settings()