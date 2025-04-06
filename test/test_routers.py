from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch

client = TestClient(app)

def test_generate_team():
    """Test para el endpoint de generación de equipos"""
    with patch('app.ai_assistant.recommendation_engine.TeamRecommender') as mock_recommender:
        mock_recommender.return_value.generate_team.return_value = {
            "formation": "4-3-3",
            "players": [],
            "total_value": 0
        }
        
        response = client.post(
            "/api/teams/generate",
            headers={"X-User-ID": "test_user"},
            json={
                "team_description": "test",
                "team_formation": "4-3-3",
                "budget": 1000000,
                "criteria": {
                    "GK": {"min_overall": 80}
                }
            }
        )
        
        assert response.status_code == 200
        assert response.json()["formation"] == "4-3-3"

def test_get_history():
    """Test para el endpoint de historial"""
    with patch('app.ai_assistant.recommendation_engine.TeamRecommender') as mock_recommender:
        mock_recommender.return_value.history.get_history.return_value = [
            {"request": {}, "response": {"formation": "4-4-2"}}
        ]
        
        response = client.get(
            "/api/teams/history",
            headers={"X-User-ID": "test_user"}
        )
        
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["formation"] == "4-4-2"

def test_invalid_formation():
    """Test para formación inválida"""
    response = client.post(
        "/api/teams/generate",
        json={
            "team_description": "test",
            "team_formation": "invalid",
            "budget": 1000000,
            "criteria": {}
        }
    )
    
    assert response.status_code == 400
    assert "La formación debe sumar" in response.json()["detail"]