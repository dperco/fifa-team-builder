from setuptools import setup, find_packages

setup(
    name="fifa_team_builder",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Lista de dependencias (opcional, pero recomendado)
        "fastapi",
        "pandas",
        "sentence-transformers",
        "faiss-cpu",  # o faiss-gpu si usas NVIDIA
    ],
)