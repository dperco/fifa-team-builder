import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from app.ai_assistant.recommendation_engine import TeamRecommender


@pytest.fixture
def sample_data():
    """Fixture con datos de jugadores de ejemplo"""
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Name': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5'],
        'BestPosition': ['GK', 'CB', 'CM', 'ST', 'RB'],
        'Overall': [80, 75, 78, 82, 77],
        'ValueEUR': [1000000, 5000000, 3000000, 7000000, 4000000],
        'Nationality': ['Arg', 'Bra', 'Esp', 'Fra', 'Eng'],
        'Potential': [85, 80, 82, 88, 81],
        'Pace': [60, 70, 75, 85, 80],
        'Shooting': [40, 50, 60, 85, 55],
        'Passing': [50, 60, 75, 70, 65],
        'Defending': [80, 85, 70, 40, 75]
    })

@pytest.fixture
def mock_embedder():
    """Mock del embedder"""
    mock = MagicMock()
    mock.encode.return_value = np.array([0.1, 0.2, 0.3])
    return mock

@pytest.fixture
def mock_index():
    """Mock del índice FAISS"""
    mock = MagicMock()
    mock.search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]]))
    return mock

@pytest.fixture
def recommender(sample_data, mock_embedder, mock_index):
    """Fixture del recomendador con datos de prueba"""
    return TeamRecommender(df=sample_data, embedder=mock_embedder, index=mock_index)

def test_gk_selection(recommender):
    """Test para selección de portero"""
    criteria = {'min_overall': 75}
    budget = 2000000
    used_ids = set()
    
    gk = recommender._select_gk(criteria, budget, used_ids)
    
    assert gk is not None
    assert gk['Position'] == 'GK'
    assert gk['Overall'] >= 75
    assert gk['ValueEUR'] <= budget
    assert "portero" in gk['SelectionReason'].lower()

def test_formation_parsing(recommender):
    """Test para parseo de formaciones"""
    assert recommender._parse_formation("4-3-3") == {'GK': 1, 'DEF': 4, 'MID': 3, 'ATT': 3}
    assert recommender._parse_formation("4-4-2") == {'GK': 1, 'DEF': 4, 'MID': 4, 'ATT': 2}
    assert not recommender._parse_formation("invalid")

def test_defender_positions(recommender):
    """Test para generación de posiciones defensivas"""
    assert recommender._get_defensive_positions("4-3-3") == ['CB', 'CB', 'FB', 'FB']
    assert recommender._get_defensive_positions("3-5-2") == ['CB', 'CB', 'FB']

def test_team_generation(recommender):
    """Test completo de generación de equipo"""
    team = recommender.generate_team(
        description="equipo prueba",
        formation="4-3-3",
        criteria={
            "GK": {"min_overall": 75},
            "DEF": {"min_pace": 65},
            "MID": {"min_passing": 70},
            "ATT": {"min_shooting": 75}
        },
        budget=10000000,
        user_id="test_user"
    )
    
    assert team['formation'] == "4-3-3"
    assert len(team['players']) > 0
    assert team['total_value'] <= 10000000
    assert 'team_analysis' in team

def test_empty_response(recommender):
    """Test para caso sin jugadores que cumplan criterios"""
    team = recommender.generate_team(
        description="equipo imposible",
        formation="4-3-3",
        criteria={
            "GK": {"min_overall": 95},  # Demasiado alto
            "DEF": {"min_pace": 95},
            "MID": {"min_passing": 95},
            "ATT": {"min_shooting": 95}
        },
        budget=100000,
        user_id="test_user"
    )
    
    assert len(team['players']) == 0
    assert "No se pudo generar" in team['team_analysis']