import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class HistoryManager:
    def __init__(self, storage_path: str = "data/team_history.json"):
        self.storage_path = Path(storage_path)
        self._ensure_storage()

    def _ensure_storage(self):
        """Crea el archivo de historial si no existe"""
        if not self.storage_path.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump([], f)

    def add_request(self, user_id: str, request_data: Dict, response_data: Dict):
        """Añade una nueva solicitud al historial"""
        try:
            history = self.get_history()
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "request": request_data,
                "response": response_data,
                "team_hash": hash(json.dumps(response_data, sort_keys=True))
            }
            
            history.append(entry)
            
            with open(self.storage_path, 'w') as f:
                json.dump(history[-100:], f)  # Mantener solo las últimas 100 entradas
            
            return entry
        except Exception as e:
            logger.error(f"Error añadiendo al historial: {str(e)}")
            return None

    def get_history(self, user_id: str = None) -> List[Dict]:
        """Obtiene el historial completo o filtrado por usuario"""
        try:
            with open(self.storage_path, 'r') as f:
                history = json.load(f)
            
            if user_id:
                return [entry for entry in history if entry.get('user_id') == user_id]
            return history
        except Exception as e:
            logger.error(f"Error leyendo historial: {str(e)}")
            return []

    def get_last_team(self, user_id: str) -> Dict:
        """Obtiene el último equipo generado por el usuario"""
        user_history = self.get_history(user_id)
        return user_history[-1] if user_history else None

    def get_similar_teams(self, team_hash: int) -> List[Dict]:
        """Busca equipos similares en el historial"""
        history = self.get_history()
        return [entry for entry in history if entry.get('team_hash') == team_hash]