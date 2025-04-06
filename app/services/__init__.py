"""
Módulo de servicios para el Asistente de Fútbol FIFA

Exporta todos los servicios disponibles para ser utilizados en la aplicación:
- Procesamiento de datos
- Generación de embeddings
- Utilidades compartidas
"""

from .data_processing import (
    load_and_preprocess_data,
    filter_by_position
    )
  
from .embeddings import (
    generate_embeddings,
    load_embeddings_index,
    
)



# Exporta todas las funciones públicamente disponibles
__all__ = [
    # Data processing
    'load_and_preprocess_data',
    'filter_by_position',
   
    
    # Embeddings
    'generate_embeddings',
    'load_embeddings_index',
]

# Versión del módulo de servicios
__version__ = '1.1.0'