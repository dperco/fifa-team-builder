from .ai_assistant.chat_processor import FIFAAssistant
from .services.data_processing import load_and_preprocess_data
from .services.embeddings import generate_embeddings
from .ai_assistant.recommendation_engine import TeamRecommender
__all__ = ['FIFAAssistant', 'load_and_preprocess_data', 'generate_embeddings', 'TeamRecommender']
__version__ = '1.0.0'