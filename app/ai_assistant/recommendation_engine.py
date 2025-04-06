
# seleccion avanzada de equipo 

import pandas as pd
import numpy as np
import faiss
from typing import Dict, List, Tuple
import logging


logger = logging.getLogger(__name__)

class TeamRecommender:
    def __init__(self, df: pd.DataFrame, embedder, index):
        self.df = self._preprocess_data(df)
        self.embedder = embedder
        self.index = index
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ['ID', 'Name', 'BestPosition', 'Overall', 'ValueEUR', 'Nationality', 
                        'Potential', 'Height', 'SprintSpeed', 'Agility', 'Dribbling',
                        'BallControl', 'Jumping', 'Interceptions', 'Marking', 'Crossing',
                        'ShortPassing', 'Positioning', 'Vision', 'Penalties', 'ShotPower',
                        'DefendingTotal', 'PhysicalityTotal', 'ShootingTotal', 'PassingTotal']
        
        # Verificar columnas
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")
        
        df = df[required_cols].copy()
        df = df.dropna(subset=['BestPosition', 'Overall'])
        df['ValueEUR'] = pd.to_numeric(df['ValueEUR'], errors='coerce').fillna(0)
        
        # Calcular scores compuestos
        df['GK_Score'] = (df['Overall'] + df['Penalties'] + df['ShotPower']) / 3
        df['CB_Score'] = (df['Potential'] + df['Height'] + df['ShootingTotal'] + 
                         df['PassingTotal'] + df['DefendingTotal'] + df['BallControl'] + 
                         df['Jumping'] + df['Interceptions'] + df['Marking']) / 9
        df['FB_Score'] = (df['Potential'] + df['ShootingTotal'] + df['PassingTotal'] + 
                         df['DefendingTotal'] + df['BallControl'] + df['Jumping'] + 
                         df['Interceptions'] + df['Marking'] + df['SprintSpeed'] + 
                         df['Agility']) / 10
        df['CM_Score'] = (df['Potential'] + df['ShootingTotal'] + df['PassingTotal'] + 
                         df['DefendingTotal'] + df['BallControl'] + df['Jumping'] + 
                         df['Interceptions'] + df['Marking'] + df['Crossing'] + 
                         df['PhysicalityTotal'] + df['ShortPassing'] + df['Positioning'] + 
                         df['Vision']) / 12
        df['CAM_CDM_Score'] = (df['Potential'] + df['ShootingTotal'] + df['PassingTotal'] + 
                              df['DefendingTotal'] + df['BallControl'] + df['Interceptions'] + 
                              df['Marking'] + df['Crossing'] + df['PhysicalityTotal'] + 
                              df['ShortPassing'] + df['Positioning'] + df['Vision'] + 
                              df['SprintSpeed'] + df['Agility'] + df['Dribbling']) / 15
        df['ST_Score'] = (df['Potential'] + df['ShootingTotal'] + df['PassingTotal'] + 
                         df['BallControl'] + df['Marking'] + df['PhysicalityTotal'] + 
                         df['ShortPassing'] + df['Positioning'] + df['Vision'] + 
                         df['SprintSpeed'] + df['Agility'] + df['Dribbling'] + 
                         df['Jumping']) / 13
        
        return df

    def generate_team(self, description: str, formation: str, criteria: Dict, budget: float) -> Dict:
        try:
            positions = self._parse_formation(formation)
            if not positions:
                return self._empty_response(formation, description)
            
            selected_players = []
            used_ids = set()
            remaining_budget = budget
            
            # 1. Seleccionar portero (GK)
            gk = self._select_gk(criteria.get('GK', {}), remaining_budget, used_ids)
            if gk is not None:
                selected_players.append(gk)
                used_ids.add(gk['ID'])
                remaining_budget -= gk['ValueEUR']
            
            # 2. Seleccionar defensores según formación
            def_players = self._select_defenders(formation, criteria.get('DEF', {}), remaining_budget, used_ids)
            selected_players.extend(def_players)
            used_ids.update([p['ID'] for p in def_players])
            remaining_budget -= sum(p['ValueEUR'] for p in def_players)
            
            # 3. Seleccionar mediocampistas según formación
            mid_players = self._select_midfielders(formation, criteria.get('MID', {}), remaining_budget, used_ids)
            selected_players.extend(mid_players)
            used_ids.update([p['ID'] for p in mid_players])
            remaining_budget -= sum(p['ValueEUR'] for p in mid_players)
            
            # 4. Seleccionar atacantes según formación
            att_players = self._select_attackers(formation, criteria.get('ATT', {}), remaining_budget, used_ids)
            selected_players.extend(att_players)
            used_ids.update([p['ID'] for p in att_players])
            remaining_budget -= sum(p['ValueEUR'] for p in att_players)
            
            return self._format_response(selected_players, formation, description)
            
        except Exception as e:
            logger.error(f"Error generando equipo: {str(e)}", exc_info=True)
            return self._empty_response(formation, description)

    def _select_gk(self, criteria: Dict, budget: float, used_ids: set) -> Dict:
        """Selecciona el mejor portero según criterios."""
        filtered = self.df[
            (self.df['BestPosition'] == 'GK') &
            (~self.df['ID'].isin(used_ids)) &
            (self.df['ValueEUR'] <= budget)
        ].copy()
        
        for attr, min_val in criteria.items():
            col = attr.replace('min_', '').capitalize()
            if col in filtered.columns:
                filtered = filtered[filtered[col] >= min_val]
        
        if filtered.empty:
            return None
            
        best_gk = filtered.nlargest(1, 'GK_Score').iloc[0]
        return {
            'ID': best_gk['ID'],
            'Name': best_gk['Name'],
            'Position': 'GK',
            'Overall': best_gk['Overall'],
            'ValueEUR': best_gk['ValueEUR'],
            'Nationality': best_gk['Nationality'],
            'SelectionReason': f"Mejor portero disponible (GK Score: {best_gk['GK_Score']:.2f}) con {best_gk['Overall']} de overall"
        }

    def _select_defenders(self, formation: str, criteria: Dict, budget: float, used_ids: set) -> List[Dict]:
        """Selecciona defensores según formación."""
        positions_needed = self._get_defensive_positions(formation)
        selected = []
        
        for pos_type in positions_needed:
            if budget <= 0:
                break
                
            if pos_type == 'CB':
                player = self._select_player(
                    pos_filter=['CB'],
                    score_col='CB_Score',
                    criteria=criteria,
                    budget=budget,
                    used_ids=used_ids,
                    pos_name='CB'
                )
                reason = f"Central (CB Score: {player['CB_Score']:.2f}) con buen potencial y habilidades defensivas"
            else:  # FB (LW/RW)
                player = self._select_player(
                    pos_filter=['LB', 'RB', 'LWB', 'RWB'],
                    score_col='FB_Score',
                    criteria=criteria,
                    budget=budget,
                    used_ids=used_ids,
                    pos_name=pos_type
                )
                reason = f"Lateral ({pos_type}, FB Score: {player['FB_Score']:.2f}) con velocidad y habilidad ofensiva/defensiva"
            
            if player is not None:
                player['SelectionReason'] = reason
                selected.append(player)
                used_ids.add(player['ID'])
                budget -= player['ValueEUR']
        
        return selected

    def _select_midfielders(self, formation: str, criteria: Dict, budget: float, used_ids: set) -> List[Dict]:
        """Selecciona mediocampistas según formación."""
        positions_needed = self._get_midfield_positions(formation)
        selected = []
        
        for pos_type in positions_needed:
            if budget <= 0:
                break
                
            if pos_type == 'CM':
                player = self._select_player(
                    pos_filter=['CM'],
                    score_col='CM_Score',
                    criteria=criteria,
                    budget=budget,
                    used_ids=used_ids,
                    pos_name='CM'
                )
                reason = f"Mediocentro (CM Score: {player['CM_Score']:.2f}) con equilibrio entre ataque y defensa"
            else:  # CAM o CDM
                player = self._select_player(
                    pos_filter=['CAM', 'CDM'],
                    score_col='CAM_CDM_Score',
                    criteria=criteria,
                    budget=budget,
                    used_ids=used_ids,
                    pos_name=pos_type
                )
                role = "Mediocentro ofensivo" if pos_type == 'CAM' else "Mediocentro defensivo"
                reason = f"{role} (CAM/CDM Score: {player['CAM_CDM_Score']:.2f}) con habilidades completas"
            
            if player is not None:
                player['SelectionReason'] = reason
                selected.append(player)
                used_ids.add(player['ID'])
                budget -= player['ValueEUR']
        
        return selected

    def _select_attackers(self, formation: str, criteria: Dict, budget: float, used_ids: set) -> List[Dict]:
        """Selecciona atacantes según formación."""
        positions_needed = self._get_attacker_positions(formation)
        selected = []
        
        for pos_type in positions_needed:
            if budget <= 0:
                break
                
            player = self._select_player(
                pos_filter=['ST', 'LW', 'RW', 'CF'],
                score_col='ST_Score',
                criteria=criteria,
                budget=budget,
                used_ids=used_ids,
                pos_name=pos_type
            )
            role = {
                'ST': "Delantero centro",
                'LW': "Extremo izquierdo", 
                'RW': "Extremo derecho",
                'CF': "Mediapunta"
            }.get(pos_type, pos_type)
            
            if player is not None:
                player['SelectionReason'] = f"{role} (ST Score: {player['ST_Score']:.2f}) con habilidades ofensivas completas"
                selected.append(player)
                used_ids.add(player['ID'])
                budget -= player['ValueEUR']
        
        return selected

    def _select_player(self, pos_filter: List[str], score_col: str, criteria: Dict, 
                      budget: float, used_ids: set, pos_name: str) -> Dict:
        """Selecciona el mejor jugador para una posición específica."""
        filtered = self.df[
            (self.df['BestPosition'].isin(pos_filter)) &
            (~self.df['ID'].isin(used_ids)) &
            (self.df['ValueEUR'] <= budget)
        ].copy()
        
        for attr, min_val in criteria.items():
            col = attr.replace('min_', '').capitalize()
            if col in filtered.columns:
                filtered = filtered[filtered[col] >= min_val]
        
        if filtered.empty:
            return None
            
        best_player = filtered.nlargest(1, score_col).iloc[0]
        return {
            'ID': best_player['ID'],
            'Name': best_player['Name'],
            'Position': pos_name,
            'Overall': best_player['Overall'],
            'ValueEUR': best_player['ValueEUR'],
            'Nationality': best_player['Nationality'],
            score_col: best_player[score_col]
        }

    def _parse_formation(self, formation: str) -> bool:
        """Valida la formación."""
        try:
            parts = [int(x) for x in formation.split('-')]
            return len(parts) in [2, 3] and sum(parts) == 10
        except:
            return False

    def _get_defensive_positions(self, formation: str) -> List[str]:
        """Devuelve las posiciones defensivas según formación."""
        parts = [int(x) for x in formation.split('-')]
        def_count = parts[0]
        
        # 2 centrales y el resto laterales
        positions = ['CB'] * min(2, def_count)
        positions.extend(['FB'] * (def_count - len(positions)))
        
        return positions

    def _get_midfield_positions(self, formation: str) -> List[str]:
        """Devuelve las posiciones de mediocampo según formación."""
        parts = [int(x) for x in formation.split('-')]
        mid_count = parts[1]
        
        if formation == '4-3-3':
            return ['CM', 'CAM', 'CDM']
        elif formation == '4-4-2':
            return ['CM', 'CM', 'RM', 'LM']
        else:
            return ['CM'] * mid_count

    def _get_attacker_positions(self, formation: str) -> List[str]:
        """Devuelve las posiciones de ataque según formación."""
        parts = [int(x) for x in formation.split('-')]
        if len(parts) == 3:
            att_count = parts[2]
        else:
            att_count = 1  # Para formaciones como 4-4-1-1
            
        if formation == '4-3-3':
            return ['ST', 'LW', 'RW']
        elif formation == '4-4-2':
            return ['ST', 'ST']
        else:
            return ['ST'] * att_count

    def _format_response(self, players: List[Dict], formation: str, description: str) -> Dict:
        """Formatea la respuesta final."""
        if not players:
            return self._empty_response(formation, description)
            
        total_value = sum(p['ValueEUR'] for p in players)
        avg_rating = sum(p['Overall'] for p in players) / len(players)
        
        formatted_players = []
        for p in players:
            formatted = {
                'id': int(p['ID']),
                'name': p['Name'],
                'position': p['Position'],
                'overall': int(p['Overall']),
                'value': float(p['ValueEUR']),
                'age': 25,  # Puedes cambiar esto si tienes la edad en tus datos
                'nationality': p['Nationality'],
                'selection_reason': p.get('SelectionReason', 'Seleccionado por rendimiento general')
            }
            formatted_players.append(formatted)
        
        return {
            'formation': formation,
            'description': description,
            'players': formatted_players,
            'total_value': float(total_value),
            'avg_rating': float(round(avg_rating, 2)),
            'team_analysis': self._generate_team_analysis(formatted_players)
        }

    def _generate_team_analysis(self, players: List[Dict]) -> str:
        """Genera un análisis del equipo seleccionado."""
        positions = {}
        for p in players:
            positions[p['position']] = positions.get(p['position'], 0) + 1
        
        analysis = "Análisis del equipo:\n"
        analysis += f"- Portero: {next((p['name'] for p in players if p['position'] == 'GK'), 'No seleccionado')}\n"
        
        defs = [p for p in players if p['position'] in ['CB', 'FB']]
        if defs:
            analysis += f"- Defensas ({len(defs)}): {', '.join(p['name'] for p in defs)}\n"
        
        mids = [p for p in players if p['position'] in ['CM', 'CAM', 'CDM', 'RM', 'LM']]
        if mids:
            analysis += f"- Mediocampistas ({len(mids)}): {', '.join(p['name'] for p in mids)}\n"
        
        atts = [p for p in players if p['position'] in ['ST', 'LW', 'RW', 'CF']]
        if atts:
            analysis += f"- Atacantes ({len(atts)}): {', '.join(p['name'] for p in atts)}\n"
        
        strengths = []
        if sum(p['overall'] for p in players if p['position'] == 'GK') / len([p for p in players if p['position'] == 'GK']) > 75:
            strengths.append("Portería sólida")
        if len([p for p in players if p['position'] in ['CB', 'FB'] and p['overall'] > 75]) >= 2:
            strengths.append("Defensa consistente")
        if len([p for p in players if p['position'] in ['CM', 'CAM', 'CDM'] and p['overall'] > 75]) >= 2:
            strengths.append("Mediocampo competitivo")
        if len([p for p in players if p['position'] in ['ST', 'LW', 'RW'] and p['overall'] > 75]) >= 2:
            strengths.append("Ataque peligroso")
        
        if strengths:
            analysis += "\nFortalezas del equipo: " + ", ".join(strengths) + "\n"
        
        return analysis

    def _empty_response(self, formation: str, description: str) -> Dict:
        return {
            'formation': formation,
            'description': description,
            'players': [],
            'total_value': 0.0,
            'avg_rating': 0.0,
            'team_analysis': 'No se pudo generar el equipo con los criterios dados'
        }