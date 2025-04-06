from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.ai_assistant.chat_processor import FIFAAssistant
import logging

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"],
    responses={404: {"description": "Not found"}}
)

logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    user_id: str
    message: str

def get_assistant():
    # Configuración inicial del asistente
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    
    # Carga datos y modelo (ajusta las rutas)
    df = pd.read_csv("data/players_21.csv")
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    return FIFAAssistant(df=df, embedder=embedder)

@router.post("")
async def chat(
    request: ChatRequest,
    assistant: FIFAAssistant = Depends(get_assistant)
):
    try:
        response = assistant.process_message(request.user_id, request.message)
        return {"response": response}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing message")