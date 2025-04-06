import re
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class IntentDetector:
    """
    Clase para detección de intenciones usando embeddings y similitud semántica
    
    Args:
        embedder (SentenceTransformer): Modelo de embeddings para texto
    """
    
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.intents = self._initialize_intents()
        self.intent_embeddings = self._generate_intent_embeddings()
    
    def _initialize_intents(self) -> Dict[str, Dict[str, Any]]:
        """Define los patrones de intención y respuestas del asistente"""
        return {
            'greeting': {
                'patterns': [
                    'hola', 'buenos días', 'buenas tardes', 
                    'hi', 'hello', 'qué tal'
                ],
                'responses': [
                    "¡Hola! ¿En qué puedo ayudarte hoy?",
                    "¡Bienvenido! ¿Cómo puedo asistirte?"
                ]
            },
            'team_creation': {
                'patterns': [
                    'quiero crear un equipo',
                    'formar un equipo ideal',
                    'armar un equipo',
                    'crear equipo de fútbol'
                ],
                'responses': [
                    "Vamos a crear tu equipo ideal. ¿Qué estilo de juego prefieres?",
                    "Perfecto, dime ¿qué estilo de juego te gustaría implementar?"
                ],
                'context_set': 'awaiting_style'
            },
            'style_description': {
                'patterns': [
                    'estilo ofensivo', 'juego defensivo',
                    'contraataque', 'posesión de balón',
                    'alta presión'
                ],
                'responses': [
                    "Entendido, prefieres un estilo {style}. ¿Qué formación te gustaría usar?"
                ],
                'context': 'awaiting_style',
                'context_set': 'awaiting_formation'
            },
            'formation_specification': {
                'patterns': [
                    '4-3-3', '4-4-2', '3-5-2',
                    'formación defensiva', 'formación ofensiva'
                ],
                'responses': [
                    "¡Buena elección! Formación {formation} para un estilo {style}."
                ],
                'context': 'awaiting_formation',
                'action': 'generate_team'
            },
            'player_inquiry': {
                'patterns': [
                    'mejor jugador para', 'quién es bueno en',
                    'recomiéndame un', 'jugador con'
                ],
                'responses': [
                    "Te recomendaría a {player} para {attribute}"
                ],
                'action': 'recommend_player'
            }
        }
    
    def _generate_intent_embeddings(self) -> Dict[str, np.ndarray]:
        """Genera embeddings para todos los patrones de intención"""
        embeddings = {}
        for intent, data in self.intents.items():
            if 'patterns' in data and data['patterns']:
                embeddings[intent] = self.embedder.encode(data['patterns'])
        return embeddings
    
    def detect_intent(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Detecta la intención en un mensaje del usuario
        
        Args:
            message (str): Mensaje de entrada del usuario
            
        Returns:
            Dict[str, Any]: Diccionario con la intención detectada y metadata
                           o None si no se detecta intención
        """
        try:
            # Preprocesamiento del mensaje
            cleaned_msg = re.sub(r'[^\w\s]', '', message.lower())
            msg_embedding = self.embedder.encode([cleaned_msg])
            
            # Búsqueda de intención más similar
            best_intent = None
            best_similarity = 0.0
            
            for intent, embeddings in self.intent_embeddings.items():
                sim = np.max(cosine_similarity(msg_embedding, embeddings))
                if sim > best_similarity and sim > 0.65:  # Umbral de similitud
                    best_similarity = sim
                    best_intent = intent
            
            if best_intent:
                return {
                    'intent': best_intent,
                    'confidence': float(best_similarity),
                    'metadata': self.intents[best_intent]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detectando intención: {str(e)}")
            raise
    
    def get_response(self, intent: str, context: Dict[str, Any] = None) -> str:
        """
        Genera una respuesta apropiada para la intención detectada
        
        Args:
            intent (str): Intención detectada
            context (Dict): Contexto actual de la conversación
            
        Returns:
            str: Respuesta generada
        """
        intent_data = self.intents.get(intent, {})
        responses = intent_data.get('responses', [])
        
        if responses:
            response = np.random.choice(responses)
            
            # Reemplazo de variables en la respuesta
            if context:
                if 'style' in context and '{style}' in response:
                    response = response.replace('{style}', context['style'])
                if 'formation' in context and '{formation}' in response:
                    response = response.replace('{formation}', context['formation'])
            
            return response
        
        return "No estoy seguro de cómo responder a eso. ¿Podrías reformularlo?"