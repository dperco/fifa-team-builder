# scripts/initialize.py


import pandas as pd
from services.data_processing import load_and_preprocess_data
from services.embeddings import generate_embeddings
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Cargando y procesando datos...")
    df = load_and_preprocess_data(settings.DATA_PATH)
    
    logger.info("Generando embeddings...")
    embeddings, index = generate_embeddings(df, settings.EMBEDDINGS_PATH)
    
    logger.info(f"Embeddings guardados en: {settings.EMBEDDINGS_PATH}")
    logger.info(f"Dimensión de los embeddings: {embeddings.shape}")
    logger.info(f"Tamaño del índice FAISS: {index.ntotal} vectores indexados")

if __name__ == "__main__":
    main()