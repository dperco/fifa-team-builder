from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from app.services.data_processing import filter_by_position
from .intent_detection import IntentDetector
from sentence_transformers import SentenceTransformer
from app.services.data_processing import filter_by_position
from app.services.embeddings import get_similar_players
logger = logging.getLogger(__name__)

class FIFAAssistant:
    def __init__(self, df: pd.DataFrame, embedder: SentenceTransformer):
        self.df = df
        self.embedder = embedder
        self.context: Dict[str, Dict[str, Any]] = {}
        self.setup_intents()
    
    def setup_intents(self):
        """Define los patrones de intención y respuestas del asistente"""
        self.intents = {
            'greeting': {
                'patterns': ['hola', 'buenos días', 'hi', 'hello', 'qué tal'],
                'responses': [
                    "¡Hola! Soy tu asistente para formar equipos de fútbol ideal. "
                    "¿En qué puedo ayudarte hoy?",
                    "¡Bienvenido! Estoy aquí para ayudarte a crear tu equipo de fútbol perfecto. "
                    "¿Por dónde empezamos?"
                ]
            },
            'team_creation': {
                'patterns': [
                    'quiero crear un equipo',
                    'formar un equipo ideal',
                    'armar un equipo',
                    'crear equipo'
                ],
                'responses': [
                    "¡Excelente decisión! Vamos a crear tu equipo ideal. "
                    "¿Podrías describirme el estilo de juego que prefieres? "
                    "(por ejemplo: 'equipo defensivo', 'juego ofensivo', 'contraataque rápido')",
                    "Perfecto, vamos a armar tu equipo. Primero, dime ¿qué estilo de juego te gustaría implementar?"
                ],
                'context_set': 'awaiting_style'
            },
            'style_description': {
                'patterns': [],
                'responses': [
                    "Entendido, prefieres un estilo de juego {style}. "
                    "¿Qué formación te gustaría usar? (ej: 4-3-3, 4-4-2, 3-5-2)",
                    "¡Buena elección! Un estilo {style} puede ser muy efectivo. "
                    "Ahora dime, ¿con qué formación quieres jugar?"
                ],
                'context': 'awaiting_style',
                'context_set': 'awaiting_formation'
            },
            'formation_received': {
                'patterns': [],
                'responses': [
                    "¡Perfecto! Formación {formation} para un estilo {style}. "
                    "Estoy generando tu equipo ideal...",
                    "Buena combinación: {formation} con estilo {style}. "
                    "Déjame encontrar los jugadores perfectos..."
                ],
                'context': 'awaiting_formation',
                'action': 'generate_team'
            },
            'player_inquiry': {
                'patterns': [
                    'mejor jugador para',
                    'quién es bueno en',
                    'recomiéndame un'
                ],
                'responses': [
                    "Basado en tu solicitud, te recomendaría a {player_name}. "
                    "Tiene un {attribute} de {value}, lo que lo hace ideal para lo que necesitas.",
                    "Para eso, {player_name} sería una excelente opción con su {attribute} de {value}."
                ],
                'action': 'recommend_player'
            },
            'thanks': {
                'patterns': ['gracias', 'thanks', 'muchas gracias'],
                'responses': [
                    "¡De nada! Estoy aquí para ayudarte. ¿Necesitas algo más?",
                    "¡Es un placer ayudarte! ¿Quieres hacer algún ajuste a tu equipo?"
                ]
            }
        }
        
        # Generar embeddings para las intenciones
        self.intent_embeddings = {}
        for intent, data in self.intents.items():
            if 'patterns' in data and data['patterns']:
                self.intent_embeddings[intent] = self.embedder.encode(data['patterns'])
    
    def detect_intent(self, message: str) -> Optional[str]:
        """Detecta la intención del mensaje usando similitud semántica"""
        cleaned_msg = re.sub(r'[^\w\s]', '', message.lower())
        msg_embedding = self.embedder.encode([cleaned_msg])
        
        best_intent = None
        best_similarity = 0.0
        
        for intent, embeddings in self.intent_embeddings.items():
            sim = cosine_similarity(msg_embedding, embeddings)
            max_sim = np.max(sim)
            
            if max_sim > best_similarity and max_sim > 0.7:  # Umbral de similitud
                best_similarity = max_sim
                best_intent = intent
        
        # Manejar casos especiales basados en contexto
        if not best_intent:
            return None
        
        return best_intent
    
    def process_message(self, user_id: str, message: str) -> str:
        """Procesa un mensaje del usuario y genera una respuesta apropiada"""
        if user_id not in self.context:
            self.context[user_id] = {}
        
        # Detectar intención
        intent = self.detect_intent(message)
        
        # Manejar mensajes basados en contexto si no se detecta intención clara
        if not intent:
            if 'awaiting_style' in self.context[user_id]:
                intent = 'style_description'
                self.context[user_id]['style'] = message
            elif 'awaiting_formation' in self.context[user_id]:
                intent = 'formation_received'
                self.context[user_id]['formation'] = message
            else:
                intent = 'unknown'
        
        # Obtener datos de la intención
        intent_data = self.intents.get(intent, {})
        responses = intent_data.get('responses', [])
        
        # Generar respuesta
        if responses:
            response = np.random.choice(responses)
            
            # Reemplazar variables en la respuesta
            if 'style' in self.context[user_id]:
                response = response.replace('{style}', self.context[user_id]['style'])
            if 'formation' in self.context[user_id]:
                response = response.replace('{formation}', self.context[user_id]['formation'])
            
            # Establecer contexto si es necesario
            if 'context_set' in intent_data:
                self.context[user_id][intent_data['context_set']] = True
            
            # Limpiar contexto si se completó una acción
            if intent == 'formation_received':
                self.context[user_id].pop('awaiting_formation', None)
                self.context[user_id].pop('awaiting_style', None)
            
            return response
        
        return "No estoy seguro de entenderte. ¿Podrías reformular tu pregunta o decirme más sobre lo que necesitas?"
    
    def is_ready_to_generate_team(self, user_id: str) -> bool:
        """Verifica si hay suficiente contexto para generar un equipo"""
        user_context = self.context.get(user_id, {})
        return 'style' in user_context and 'formation' in user_context
    
    def generate_team_from_context(self, user_id: str) -> Dict[str, Any]:
        """Genera un equipo basado en el contexto acumulado"""
        if user_id not in self.context:
            return {"error": "No hay contexto para este usuario"}
        
        ctx = self.context[user_id]
        
        if 'style' not in ctx or 'formation' not in ctx:
            return {"error": "Falta información para generar el equipo"}
        
        criteria = self._suggest_criteria(ctx['style'], ctx['formation'])
        
        team_request = {
            "team_description": ctx['style'],
            "team_formation": ctx['formation'],
            "criteria": criteria
        }
        
        # Aquí integraríamos con la lógica de generación de equipos
        # Por ahora devolvemos un mock
        return self._mock_team_generation(team_request)
    
    def _suggest_criteria(self, style: str, formation: str) -> Dict[str, Dict[str, int]]:
        """Sugiere criterios basados en estilo y formación"""
        criteria = {
            "Goalkeeper": {"goalkeeping_reflexes": 75, "goalkeeping_positioning": 70},
            "Defender": {},
            "Midfielder": {},
            "Forward": {}
        }
        
        style = style.lower()
        
        # Ajustar según estilo
        if "ofensivo" in style or "ataque" in style:
            criteria["Defender"].update({
                "attacking_short_passing": 75,
                "skill_ball_control": 70
            })
            criteria["Midfielder"].update({
                "skill_long_passing": 80,
                "mentality_vision": 80,
                "attacking_short_passing": 85
            })
            criteria["Forward"].update({
                "movement_sprint_speed": 80,
                "attacking_finishing": 80,
                "movement_acceleration": 80
            })
        elif "defensivo" in style or "defensa" in style:
            criteria["Defender"].update({
                "defending_standing_tackle": 85,
                "power_strength": 80,
                "mentality_interceptions": 80
            })
            criteria["Midfielder"].update({
                "defending_standing_tackle": 75,
                "power_stamina": 80,
                "mentality_aggression": 75
            })
            criteria["Forward"].update({
                "mentality_work_rate": 80,
                "power_strength": 75
            })
        elif "posesión" in style or "control" in style:
            criteria["Defender"].update({
                "skill_ball_control": 80,
                "attacking_short_passing": 85
            })
            criteria["Midfielder"].update({
                "skill_ball_control": 85,
                "attacking_short_passing": 90,
                "mentality_composure": 85
            })
            criteria["Forward"].update({
                "skill_dribbling": 85,
                "skill_ball_control": 85
            })
        
        # Ajustar según formación
        if "4-3-3" in formation:
            criteria["Midfielder"].update({
                "power_stamina": 80,
                "mentality_work_rate": 75
            })
            criteria["Forward"].update({
                "movement_sprint_speed": 85
            })
        elif "4-4-2" in formation:
            criteria["Midfielder"].update({
                "mentality_work_rate": 80,
                "power_stamina": 85
            })
        elif "3-5-2" in formation:
            criteria["Defender"].update({
                "mentality_positioning": 80,
                "power_strength": 85
            })
            criteria["Midfielder"].update({
                "mentality_work_rate": 85
            })
        
        return criteria
    
    def _mock_team_generation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generación mock de equipo para demostración"""
        # En una implementación real, esto usaría el dataset real
        mock_players = {
            "Goalkeeper": {
                "name": "Alisson Becker",
                "reason": "Excelentes reflejos (89) y posicionamiento (88), ideal para mantener la portería a cero."
            },
            "Defender": {
                "name": "Virgil van Dijk",
                "reason": "Fuerza física (86) y capacidad de anticipación (87), clave en la defensa."
            },
            "Midfielder": {
                "name": "Kevin De Bruyne",
                "reason": "Visión de juego (94) y precisión en pases (93), esencial para crear jugadas."
            },
            "Forward": {
                "name": "Kylian Mbappé",
                "reason": "Velocidad (96) y definición (90), letal en contraataques."
            }
        }
        
        return {
            "team_description": request["team_description"],
            "team_formation": request["team_formation"],
            **mock_players
        }