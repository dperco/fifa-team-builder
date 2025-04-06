import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import os
import logging
from typing import Tuple, Dict
from app.services.data_processing import load_and_preprocess_data
logger = logging.getLogger(__name__)

def generate_embeddings(df: pd.DataFrame, save_path: str) -> Tuple[np.ndarray, faiss.Index]:
    """Genera embeddings para los jugadores y crea índice FAISS"""
    try:
        logger.info("Generando embeddings para los jugadores...")
        embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Verificación exhaustiva de columnas
        required_columns = {
            'name': ['short_name', 'name', 'Name'],
            'age': ['age', 'Age'],
            'nationality': ['nationality', 'Nationality'],
            'positions': ['player_positions', 'Positions', 'Position'],
            'overall': ['overall', 'Overall']
        }
        
        # Encontrar los nombres reales de las columnas
        col_mapping = {}
        for key, alternatives in required_columns.items():
            for alt in alternatives:
                if alt in df.columns:
                    col_mapping[key] = alt
                    break
            else:
                available = [c for c in df.columns if c.lower() == key.lower()]
                if available:
                    col_mapping[key] = available[0]
                else:
                    raise KeyError(f"No se encontró ninguna columna compatible para: {alternatives}")
        
        logger.info(f"Columnas detectadas: {col_mapping}")
        
        # Crear descripción textual
        df['player_description'] = (
            df[col_mapping['name']].astype(str) + ' is a ' + 
            df[col_mapping['age']].astype(str) + ' years old ' +
            df[col_mapping['positions']].astype(str) + ' from ' + 
            df[col_mapping['nationality']].astype(str) + '. ' +
            'Overall rating: ' + df[col_mapping['overall']].astype(str)
        )
        
        # Generar embeddings
        embeddings = embedder.encode(df['player_description'].tolist())
        
        # Crear índice FAISS
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Guardar
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        faiss.write_index(index, save_path)
        
        logger.info(f"Embeddings generados correctamente. Dimensiones: {embeddings.shape}")
        return embeddings, index
        
    except Exception as e:
        logger.error(f"Error generando embeddings: {str(e)}")
        raise

def load_embeddings_index(filepath: str) -> faiss.Index:
    """Carga el índice FAISS desde disco"""
    try:
        logger.info(f"Cargando índice FAISS desde {filepath}")
        return faiss.read_index(filepath)
    except Exception as e:
        logger.error(f"Error cargando índice: {str(e)}")
        raise
def get_similar_players(
    team_embedding: np.ndarray,
    players_df: pd.DataFrame,
    index: faiss.Index,
    criteria: Dict[str, int],
    embedder: SentenceTransformer,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Encuentra jugadores similares a los criterios dados usando búsqueda semántica.
    """
    try:
        # 1. Verificar y normalizar nombres de columnas
        rating_col = 'Overall' if 'Overall' in players_df.columns else 'overall'
        
        # 2. Filtrar jugadores por criterios mínimos
        for attr, min_value in criteria.items():
            # Buscar el nombre real de la columna (case insensitive)
            attr_cols = [col for col in players_df.columns if col.lower() == attr.lower()]
            if attr_cols:
                players_df = players_df[players_df[attr_cols[0]] >= min_value]
        
        if players_df.empty:
            return pd.DataFrame()
        
        # 3. Obtener índices de los jugadores filtrados
        player_indices = players_df.index.values
        
        # 4. Buscar jugadores similares en el índice FAISS
        distances, indices = index.search(np.array([team_embedding]), top_k)
        
        # 5. Filtrar jugadores que cumplen los criterios
        similar_indices = [i for i in indices[0] if i in player_indices]
        similar_players = players_df.loc[similar_indices]
        
        # 6. Ordenar por puntuación general (usando el nombre de columna correcto)
        return similar_players.sort_values(rating_col, ascending=False).head(top_k)
        
    except Exception as e:
        logger.error(f"Error en búsqueda de jugadores similares: {str(e)}", exc_info=True)
        raise