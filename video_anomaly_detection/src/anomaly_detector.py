"""
Módulo de Detecção de Anomalias
Identifica comportamentos atípicos e eventos anormais
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque, defaultdict
from scipy.spatial.distance import euclidean
import time


class AnomalyDetector:
    """Detector de anomalias em vídeos de vigilância"""
    
    # Definição explícita de anomalias detectadas
    ANOMALY_TYPES = {
        'MOVIMENTO_SUBITO': {
            'description': 'Aceleração repentina ou movimento brusco',
            'severity': 'MEDIA'
        },
        'VELOCIDADE_ANORMAL': {
            'description': 'Velocidade muito acima do esperado',
            'severity': 'ALTA'
        },
        'PARADA_PROLONGADA': {
            'description': 'Pessoa parada por tempo excessivo',
            'severity': 'BAIXA'
        },
        'AGLOMERACAO': {
            'description': 'Múltiplas pessoas em área pequena',
            'severity': 'MEDIA'
        },
        'MOVIMENTO_REVERSO': {
            'description': 'Pessoa retornando ao ponto de origem',
            'severity': 'BAIXA'
        },
        'OBJETO_ABANDONADO': {
            'description': 'Objeto deixado em local por tempo prolongado',
            'severity': 'ALTA'
        },
        'DIRECAO_PROIBIDA': {
            'description': 'Movimento em direção não esperada',
            'severity': 'MEDIA'
        }
    }
    
    SEVERITY_LEVELS = ['BAIXA', 'MEDIA', 'ALTA']
    
    def __init__(self, frame_rate: float = 30.0):
        """
        Inicializa detector de anomalias
        
        Args:
            frame_rate: Taxa de frames do vídeo
        """
        self.frame_rate = frame_rate
        self.anomalies = []  # Lista de todas as anomalias detectadas
        self.track_states = {}  # Estado de cada track
        
        # Thresholds para detecção
        self.thresholds = {
            'sudden_acceleration': 10.0,   # pixels/frame²
            'high_speed': 8.0,             # pixels/frame
            'stopped_duration': 5.0,       # segundos
            'crowding_distance': 80.0,     # pixels
            'crowding_count': 3,           # pessoas
            'return_threshold': 50.0,      # pixels do ponto inicial
            'object_abandoned_time': 10.0, # segundos
        }
        
        # Histórico de velocidades
        self.velocity_history = {}
        self.history_size = 10
        
    def detect(self, tracks: List[Dict], activities: Dict[int, str], 
               frame_number: int, timestamp: float) -> List[Dict]:
        """
        Detecta anomalias no frame atual
        
        Args:
            tracks: Lista de tracks ativos
            activities: Atividades classificadas {track_id: atividade}
            frame_number: Número do frame
            timestamp: Timestamp do vídeo em segundos
            
        Returns:
            Lista de anomalias detectadas neste frame
        """
        frame_anomalies = []
        
        # Inicializar estado dos tracks
        self._update_track_states(tracks, timestamp)
        
        for track in tracks:
            track_id = track['id']
            
            # 1. Verificar movimento súbito
            if self._check_sudden_movement(track_id):
                frame_anomalies.append(self._create_anomaly(
                    'MOVIMENTO_SUBITO', track, frame_number, timestamp
                ))
            
            # 2. Verificar velocidade anormal
            if self._check_high_speed(track):
                frame_anomalies.append(self._create_anomaly(
                    'VELOCIDADE_ANORMAL', track, frame_number, timestamp
                ))
            
            # 3. Verificar parada prolongada
            if self._check_prolonged_stop(track_id, activities.get(track_id)):
                frame_anomalies.append(self._create_anomaly(
                    'PARADA_PROLONGADA', track, frame_number, timestamp
                ))
            
            # 4. Verificar movimento reverso
            if self._check_reverse_movement(track_id):
                frame_anomalies.append(self._create_anomaly(
                    'MOVIMENTO_REVERSO', track, frame_number, timestamp
                ))
            
            # 5. Verificar objeto abandonado
            if self._check_abandoned_object(track, activities.get(track_id)):
                frame_anomalies.append(self._create_anomaly(
                    'OBJETO_ABANDONADO', track, frame_number, timestamp
                ))
        
        # 6. Verificar aglomeração
        crowding_anomalies = self._check_crowding(tracks, frame_number, timestamp)
        frame_anomalies.extend(crowding_anomalies)
        
        # Adicionar às anomalias totais
        self.anomalies.extend(frame_anomalies)
        
        return frame_anomalies
    
    def _check_sudden_movement(self, track_id: int) -> bool:
        """Detecta aceleração repentina"""
        if track_id not in self.velocity_history:
            return False
        
        velocities = list(self.velocity_history[track_id])
        if len(velocities) < 3:
            return False
        
        # Calcular acelerações
        accelerations = []
        for i in range(len(velocities) - 1):
            v1 = velocities[i]
            v2 = velocities[i + 1]
            accel = np.linalg.norm(v2 - v1)
            accelerations.append(accel)
        
        # Verificar se há aceleração súbita
        if accelerations:
            max_accel = max(accelerations)
            return max_accel > self.thresholds['sudden_acceleration']
        
        return False
    
    def _check_high_speed(self, track: Dict) -> bool:
        """Detecta velocidade anormalmente alta"""
        velocity = track.get('velocity', np.array([0.0, 0.0]))
        speed = np.linalg.norm(velocity)
        return speed > self.thresholds['high_speed']
    
    def _check_prolonged_stop(self, track_id: int, activity: Optional[str]) -> bool:
        """Detecta parada prolongada"""
        if activity != 'PARADO':
            return False
        
        if track_id not in self.track_states:
            return False
        
        state = self.track_states[track_id]
        if 'stopped_since' not in state:
            return False
        
        stopped_duration = time.time() - state['stopped_since']
        return stopped_duration > self.thresholds['stopped_duration']
    
    def _check_reverse_movement(self, track_id: int) -> bool:
        """Detecta movimento de retorno ao ponto inicial"""
        if track_id not in self.track_states:
            return False
        
        state = self.track_states[track_id]
        if 'initial_position' not in state or 'current_position' not in state:
            return False
        
        # Verificar se voltou próximo ao ponto inicial
        initial = np.array(state['initial_position'])
        current = np.array(state['current_position'])
        
        distance = np.linalg.norm(current - initial)
        
        # Só considera reverso se já se afastou e voltou
        if 'max_distance' in state:
            return (distance < self.thresholds['return_threshold'] and 
                   state['max_distance'] > self.thresholds['return_threshold'] * 2)
        
        return False
    
    def _check_abandoned_object(self, track: Dict, activity: Optional[str]) -> bool:
        """Detecta objeto abandonado (não é pessoa e está parado)"""
        # Apenas para objetos, não pessoas
        if track['class_id'] == 0:  # 0 = person
            return False
        
        if activity != 'PARADO':
            return False
        
        track_id = track['id']
        if track_id not in self.track_states:
            return False
        
        state = self.track_states[track_id]
        if 'stopped_since' not in state:
            return False
        
        stopped_duration = time.time() - state['stopped_since']
        return stopped_duration > self.thresholds['object_abandoned_time']
    
    def _check_crowding(self, tracks: List[Dict], frame_number: int, 
                       timestamp: float) -> List[Dict]:
        """Detecta aglomerações"""
        anomalies = []
        
        # Filtrar apenas pessoas
        people = [t for t in tracks if t['class_id'] == 0]
        
        if len(people) < self.thresholds['crowding_count']:
            return anomalies
        
        # Verificar grupos próximos
        checked = set()
        
        for i, person1 in enumerate(people):
            if person1['id'] in checked:
                continue
            
            center1 = self._get_center(person1['bbox'])
            nearby = [person1]
            
            for j, person2 in enumerate(people):
                if i == j or person2['id'] in checked:
                    continue
                
                center2 = self._get_center(person2['bbox'])
                distance = euclidean(center1, center2)
                
                if distance < self.thresholds['crowding_distance']:
                    nearby.append(person2)
            
            if len(nearby) >= self.thresholds['crowding_count']:
                # Marcar todos como verificados
                for person in nearby:
                    checked.add(person['id'])
                
                # Criar anomalia de aglomeração
                center = np.mean([self._get_center(p['bbox']) for p in nearby], axis=0)
                
                anomaly = {
                    'type': 'AGLOMERACAO',
                    'description': self.ANOMALY_TYPES['AGLOMERACAO']['description'],
                    'severity': self.ANOMALY_TYPES['AGLOMERACAO']['severity'],
                    'frame': frame_number,
                    'timestamp': timestamp,
                    'location': tuple(center),
                    'involved_tracks': [p['id'] for p in nearby],
                    'count': len(nearby)
                }
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _update_track_states(self, tracks: List[Dict], timestamp: float):
        """Atualiza estado de cada track"""
        current_ids = {track['id'] for track in tracks}
        
        for track in tracks:
            track_id = track['id']
            velocity = track.get('velocity', np.array([0.0, 0.0]))
            center = self._get_center(track['bbox'])
            
            # Atualizar histórico de velocidades
            if track_id not in self.velocity_history:
                self.velocity_history[track_id] = deque(maxlen=self.history_size)
            self.velocity_history[track_id].append(velocity)
            
            # Inicializar estado
            if track_id not in self.track_states:
                self.track_states[track_id] = {
                    'initial_position': center,
                    'max_distance': 0.0,
                    'first_seen': timestamp
                }
            
            state = self.track_states[track_id]
            state['current_position'] = center
            state['last_seen'] = timestamp
            
            # Atualizar distância máxima
            initial = np.array(state['initial_position'])
            distance = np.linalg.norm(np.array(center) - initial)
            state['max_distance'] = max(state['max_distance'], distance)
            
            # Verificar se está parado
            speed = np.linalg.norm(velocity)
            if speed < 2.0:  # threshold para "parado"
                if 'stopped_since' not in state:
                    state['stopped_since'] = timestamp
            else:
                if 'stopped_since' in state:
                    del state['stopped_since']
        
        # Limpar estados de tracks que não existem mais
        to_remove = [tid for tid in self.track_states if tid not in current_ids]
        for tid in to_remove:
            if timestamp - self.track_states[tid].get('last_seen', 0) > 5.0:
                del self.track_states[tid]
                if tid in self.velocity_history:
                    del self.velocity_history[tid]
    
    def _create_anomaly(self, anomaly_type: str, track: Dict, 
                       frame_number: int, timestamp: float) -> Dict:
        """Cria registro de anomalia"""
        info = self.ANOMALY_TYPES[anomaly_type]
        center = self._get_center(track['bbox'])
        
        return {
            'type': anomaly_type,
            'description': info['description'],
            'severity': info['severity'],
            'frame': frame_number,
            'timestamp': timestamp,
            'track_id': track['id'],
            'location': tuple(center),
            'bbox': track['bbox']
        }
    
    @staticmethod
    def _get_center(bbox: List[float]) -> Tuple[float, float]:
        """Calcula centro do bbox"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas de anomalias"""
        stats = {
            'total_anomalies': len(self.anomalies),
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int),
            'timeline': []
        }
        
        for anomaly in self.anomalies:
            stats['by_type'][anomaly['type']] += 1
            stats['by_severity'][anomaly['severity']] += 1
            
            stats['timeline'].append({
                'timestamp': anomaly['timestamp'],
                'type': anomaly['type'],
                'severity': anomaly['severity']
            })
        
        return dict(stats)
    
    def get_severity_color(self, severity: str) -> Tuple[int, int, int]:
        """Retorna cor BGR para cada nível de severidade"""
        colors = {
            'BAIXA': (0, 255, 0),    # Verde
            'MEDIA': (0, 255, 255),  # Amarelo
            'ALTA': (0, 0, 255)      # Vermelho
        }
        return colors.get(severity, (255, 255, 255))


if __name__ == "__main__":
    # Teste do detector
    detector = AnomalyDetector()
    
    print("Tipos de anomalias detectadas:")
    for anom_type, info in detector.ANOMALY_TYPES.items():
        print(f"  {anom_type}:")
        print(f"    - {info['description']}")
        print(f"    - Severidade: {info['severity']}")