"""
Módulo de Classificação de Atividades
Analisa padrões de movimento para classificar atividades
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import deque
from scipy.spatial.distance import euclidean


class ActivityClassifier:
    """Classificador de atividades baseado em análise de movimento"""
    
    # Definição explícita das atividades detectadas
    ACTIVITIES = {
        'PARADO': 'Pessoa ou objeto sem movimento significativo',
        'CAMINHANDO': 'Movimento moderado em direção consistente',
        'CORRENDO': 'Movimento rápido com alta velocidade',
        'INTERAGINDO': 'Proximidade entre múltiplas pessoas',
        'COMPORTAMENTO_ERRATICO': 'Mudanças bruscas de direção ou movimento irregular'
    }
    
    def __init__(self, frame_rate: float = 30.0):
        """
        Inicializa classificador
        
        Args:
            frame_rate: Taxa de frames do vídeo (fps)
        """
        self.frame_rate = frame_rate
        self.activity_history = {}  # {track_id: deque de atividades}
        self.position_history = {}  # {track_id: deque de posições}
        self.history_size = 30
        
        # Thresholds calibrados
        self.thresholds = {
            'stopped_speed': 2.0,        # pixels/frame
            'walking_speed': 5.0,        # pixels/frame
            'running_speed': 8.0,        # pixels/frame
            'interaction_distance': 100.0,  # pixels
            'direction_change': 45.0,    # graus
            'erratic_changes': 3         # mudanças em janela
        }
    
    def classify(self, tracks: List[Dict], frame_number: int) -> Dict[int, str]:
        """
        Classifica atividade de cada track
        
        Args:
            tracks: Lista de tracks ativos
            frame_number: Número do frame atual
            
        Returns:
            Dicionário {track_id: atividade}
        """
        activities = {}
        
        # Atualizar histórico de posições
        self._update_position_history(tracks)
        
        for track in tracks:
            track_id = track['id']
            
            # Classificar atividade individual
            activity = self._classify_individual_activity(track)
            
            # Verificar interações
            if activity != 'INTERAGINDO':
                interaction = self._check_interactions(track, tracks)
                if interaction:
                    activity = 'INTERAGINDO'
            
            activities[track_id] = activity
            
            # Atualizar histórico
            if track_id not in self.activity_history:
                self.activity_history[track_id] = deque(maxlen=self.history_size)
            self.activity_history[track_id].append(activity)
        
        return activities
    
    def _classify_individual_activity(self, track: Dict) -> str:
        """Classifica atividade de um track individual"""
        track_id = track['id']
        velocity = track.get('velocity', np.array([0.0, 0.0]))
        speed = np.linalg.norm(velocity)
        
        # Verificar comportamento errático primeiro
        if self._is_erratic_behavior(track_id):
            return 'COMPORTAMENTO_ERRATICO'
        
        # Classificar baseado em velocidade
        if speed < self.thresholds['stopped_speed']:
            return 'PARADO'
        elif speed < self.thresholds['walking_speed']:
            return 'CAMINHANDO'
        elif speed >= self.thresholds['running_speed']:
            return 'CORRENDO'
        else:
            return 'CAMINHANDO'
    
    def _is_erratic_behavior(self, track_id: int) -> bool:
        """Detecta comportamento errático baseado em mudanças de direção"""
        if track_id not in self.position_history:
            return False
        
        positions = list(self.position_history[track_id])
        if len(positions) < 10:
            return False
        
        # Calcular mudanças de direção
        direction_changes = 0
        window_size = 5
        
        for i in range(len(positions) - window_size * 2):
            # Direção na primeira janela
            dir1 = self._calculate_direction(
                positions[i:i + window_size]
            )
            
            # Direção na segunda janela
            dir2 = self._calculate_direction(
                positions[i + window_size:i + window_size * 2]
            )
            
            if dir1 is not None and dir2 is not None:
                # Calcular diferença angular
                angle_diff = abs(self._angle_difference(dir1, dir2))
                
                if angle_diff > self.thresholds['direction_change']:
                    direction_changes += 1
        
        return direction_changes >= self.thresholds['erratic_changes']
    
    def _check_interactions(self, track: Dict, all_tracks: List[Dict]) -> bool:
        """Verifica se track está interagindo com outros"""
        track_center = self._get_center(track['bbox'])
        
        for other_track in all_tracks:
            if other_track['id'] == track['id']:
                continue
            
            # Apenas verificar interações pessoa-pessoa
            if track['class_id'] != 0 or other_track['class_id'] != 0:
                continue
            
            other_center = self._get_center(other_track['bbox'])
            distance = euclidean(track_center, other_center)
            
            if distance < self.thresholds['interaction_distance']:
                return True
        
        return False
    
    def _update_position_history(self, tracks: List[Dict]):
        """Atualiza histórico de posições"""
        for track in tracks:
            track_id = track['id']
            center = self._get_center(track['bbox'])
            
            if track_id not in self.position_history:
                self.position_history[track_id] = deque(maxlen=self.history_size)
            
            self.position_history[track_id].append(center)
    
    @staticmethod
    def _get_center(bbox: List[float]) -> Tuple[float, float]:
        """Calcula centro do bbox"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @staticmethod
    def _calculate_direction(positions: List[Tuple[float, float]]) -> float:
        """Calcula direção média (em graus) de uma sequência de posições"""
        if len(positions) < 2:
            return None
        
        # Vetor de deslocamento total
        start = np.array(positions[0])
        end = np.array(positions[-1])
        displacement = end - start
        
        if np.linalg.norm(displacement) < 1e-6:
            return None
        
        # Ângulo em graus
        angle = np.degrees(np.arctan2(displacement[1], displacement[0]))
        return angle
    
    @staticmethod
    def _angle_difference(angle1: float, angle2: float) -> float:
        """Calcula diferença entre dois ângulos (0-180 graus)"""
        diff = abs(angle1 - angle2)
        if diff > 180:
            diff = 360 - diff
        return diff
    
    def get_activity_statistics(self) -> Dict:
        """Retorna estatísticas de atividades ao longo do tempo"""
        stats = {
            'total_tracks': len(self.activity_history),
            'activity_counts': {},
            'track_activities': {}
        }
        
        # Contar ocorrências de cada atividade
        for activity in self.ACTIVITIES.keys():
            stats['activity_counts'][activity] = 0
        
        # Processar histórico
        for track_id, activities in self.activity_history.items():
            activity_list = list(activities)
            
            # Atividade mais frequente para este track
            if activity_list:
                most_common = max(set(activity_list), key=activity_list.count)
                stats['track_activities'][track_id] = {
                    'most_common': most_common,
                    'distribution': {
                        activity: activity_list.count(activity) / len(activity_list)
                        for activity in set(activity_list)
                    }
                }
                
                # Contar para estatísticas gerais
                for activity in activity_list:
                    stats['activity_counts'][activity] += 1
        
        return stats
    
    def get_activity_description(self, activity: str) -> str:
        """Retorna descrição de uma atividade"""
        return self.ACTIVITIES.get(activity, "Atividade desconhecida")


if __name__ == "__main__":
    # Teste do classificador
    classifier = ActivityClassifier()
    
    # Simular tracks
    test_tracks = [
        {
            'id': 1,
            'bbox': [100, 100, 150, 200],
            'class_id': 0,
            'velocity': np.array([1.0, 0.5])
        },
        {
            'id': 2,
            'bbox': [110, 105, 160, 205],
            'class_id': 0,
            'velocity': np.array([0.8, 0.3])
        }
    ]
    
    activities = classifier.classify(test_tracks, 0)
    print("Atividades detectadas:")
    for track_id, activity in activities.items():
        print(f"  Track {track_id}: {activity}")
        print(f"    Descrição: {classifier.get_activity_description(activity)}")