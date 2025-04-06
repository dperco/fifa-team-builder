import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """Carga y preprocesa el dataset de FIFA 21"""
    try:
        logger.info(f"Cargando datos desde {filepath}")
        df = pd.read_csv(filepath, low_memory=False)
        
        # Verificar nombres alternativos de columnas
        positions_col = 'player_positions' if 'player_positions' in df.columns else 'Positions'
        overall_col = 'overall' if 'overall' in df.columns else 'Overall'
        
        # Limpieza básica
        df = df.dropna(subset=[positions_col, overall_col])
        df = df[df[positions_col].notna()]
        
        # Normalizar nombres de columnas
        df.columns = df.columns.str.lower()
        
        # Crear columna de posición principal (usando el nombre correcto)
        positions_col = positions_col.lower()
        df['main_position'] = df[positions_col].str.split(',').str[0].str.strip()
        
        # Clasificación de posiciones
        position_mapping = {
            'GK': 'Goalkeeper',
            'CB': 'Defender', 'RB': 'Defender', 'LB': 'Defender', 
            'RWB': 'Defender', 'LWB': 'Defender',
            'CDM': 'Midfielder', 'CM': 'Midfielder', 'CAM': 'Midfielder', 
            'RM': 'Midfielder', 'LM': 'Midfielder',
            'RW': 'Forward', 'LW': 'Forward', 'CF': 'Forward', 'ST': 'Forward'
        }
        
        df['position_group'] = df['main_position'].map(position_mapping)
        df['position_group'] = df['position_group'].fillna('Other')
        
        # Convertir columnas numéricas
        numeric_cols = [
            'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
            'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
            'attacking_short_passing', 'attacking_volleys', 'skill_dribbling',
            'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
            'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
            'movement_reactions', 'movement_balance', 'power_shot_power',
            'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
            'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
            'mentality_vision', 'mentality_penalties', 'mentality_composure',
            'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle',
            'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
            'goalkeeping_positioning', 'goalkeeping_reflexes'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Datos cargados correctamente. {len(df)} jugadores disponibles")
        return df
    
    except Exception as e:
        logger.error(f"Error procesando datos: {str(e)}")
        raise

def filter_by_position(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Filtra jugadores por grupo de posición"""
    position_groups = {
        'Goalkeeper': ['GK'],
        'Defender': ['CB', 'RB', 'LB', 'RWB', 'LWB'],
        'Midfielder': ['CDM', 'CM', 'CAM', 'RM', 'LM'],
        'Forward': ['RW', 'LW', 'CF', 'ST']
    }
    
    if position not in position_groups:
        return pd.DataFrame()
    
    positions = position_groups[position]
    return df[df['main_position'].isin(positions)].copy()