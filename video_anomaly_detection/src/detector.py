"""
Módulo de Detecção e Rastreamento de Objetos
Utiliza YOLOv8 para detecção e algoritmo de tracking customizado
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional
import time


class ObjectTracker:
    """Rastreador de objetos com ID único e histórico de trajetória"""
    
    def __init__(self, max_history: int = 30):
        self.tracks = {}  # {track_id: Track}
        self.next_id = 1
        self.max_history = max_history
        self.max_age = 30  # frames sem detecção antes de remover
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Atualiza tracks com novas detecções
        
        Args:
            detections: Lista de detecções [{bbox, confidence, class_id}]
            
        Returns:
            Lista de tracks ativos com IDs
        """
        if not self.tracks:
            # Primeira iteração - criar novos tracks
            for det in detections:
                self._create_track(det)
        else:
            # Associar detecções a tracks existentes
            self._associate_detections(detections)
        
        # Remover tracks antigos
        self._cleanup_tracks()
        
        # Retornar tracks ativos
        return self._get_active_tracks()
    
    def _create_track(self, detection: Dict) -> int:
        """Cria um novo track"""
        track_id = self.next_id
        self.next_id += 1
        
        self.tracks[track_id] = {
            'id': track_id,
            'bbox': detection['bbox'],
            'class_id': detection['class_id'],
            'confidence': detection['confidence'],
            'history': deque([self._bbox_center(detection['bbox'])], maxlen=self.max_history),
            'age': 0,
            'hits': 1,
            'velocity': np.array([0.0, 0.0]),
            'last_seen': time.time()
        }
        
        return track_id
    
    def _associate_detections(self, detections: List[Dict]):
        """Associa detecções a tracks existentes usando IoU"""
        if not detections:
            # Incrementar idade de todos os tracks
            for track in self.tracks.values():
                track['age'] += 1
            return
        
        # Calcular matriz de custos (1 - IoU)
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(detections), len(track_ids)))
        
        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                iou = self._calculate_iou(det['bbox'], track['bbox'])
                cost_matrix[i, j] = 1 - iou
        
        # Associação simples: menor custo
        matched_tracks = set()
        matched_detections = set()
        
        # Ordenar por menor custo
        flat_costs = []
        for i in range(len(detections)):
            for j in range(len(track_ids)):
                flat_costs.append((cost_matrix[i, j], i, j))
        flat_costs.sort()
        
        for cost, det_idx, track_idx in flat_costs:
            if det_idx in matched_detections or track_idx in matched_tracks:
                continue
            if cost < 0.7:  # Threshold de IoU mínimo
                track_id = track_ids[track_idx]
                self._update_track(track_id, detections[det_idx])
                matched_tracks.add(track_idx)
                matched_detections.add(det_idx)
        
        # Criar novos tracks para detecções não associadas
        for i, det in enumerate(detections):
            if i not in matched_detections:
                self._create_track(det)
        
        # Incrementar idade dos tracks não associados
        for j, track_id in enumerate(track_ids):
            if j not in matched_tracks:
                self.tracks[track_id]['age'] += 1
    
    def _update_track(self, track_id: int, detection: Dict):
        """Atualiza um track existente"""
        track = self.tracks[track_id]
        
        # Calcular velocidade
        old_center = self._bbox_center(track['bbox'])
        new_center = self._bbox_center(detection['bbox'])
        velocity = np.array(new_center) - np.array(old_center)
        
        # Atualizar track
        track['bbox'] = detection['bbox']
        track['confidence'] = detection['confidence']
        track['history'].append(new_center)
        track['age'] = 0
        track['hits'] += 1
        track['velocity'] = velocity
        track['last_seen'] = time.time()
    
    def _cleanup_tracks(self):
        """Remove tracks muito antigos"""
        to_remove = []
        for track_id, track in self.tracks.items():
            if track['age'] > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def _get_active_tracks(self) -> List[Dict]:
        """Retorna apenas tracks ativos"""
        active = []
        for track in self.tracks.values():
            if track['hits'] >= 3:  # Mínimo de hits para considerar válido
                active.append(track)
        return active
    
    @staticmethod
    def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
        """Calcula centro do bbox"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @staticmethod
    def _calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Calcula Intersection over Union entre dois bboxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calcular interseção
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calcular união
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class ObjectDetector:
    """Detector de objetos usando YOLOv8"""
    
    def __init__(self, model_name: str = 'yolov8n.pt', confidence: float = 0.5):
        """
        Inicializa detector
        
        Args:
            model_name: Nome do modelo YOLO
            confidence: Confiança mínima para detecção
        """
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.tracker = ObjectTracker()
        
        # Classes COCO de interesse (pessoas e objetos comuns)
        self.target_classes = [0, 24, 26, 28, 39, 41, 67]  # person, backpack, handbag, suitcase, bottle, cup, cell phone
        self.class_names = self.model.names
        
    def detect_and_track(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detecta e rastreia objetos no frame
        
        Args:
            frame: Frame de vídeo (BGR)
            
        Returns:
            frame_annotated: Frame com anotações
            tracks: Lista de tracks detectados
        """
        # Detecção
        results = self.model(frame, conf=self.confidence, classes=self.target_classes, verbose=False)
        
        # Extrair detecções
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': self.class_names[cls]
                })
        
        # Rastreamento
        tracks = self.tracker.update(detections)
        
        # Anotar frame
        frame_annotated = self._annotate_frame(frame.copy(), tracks)
        
        return frame_annotated, tracks
    
    def _annotate_frame(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """Anota frame com bounding boxes e IDs"""
        for track in tracks:
            x1, y1, x2, y2 = map(int, track['bbox'])
            track_id = track['id']
            class_name = self.class_names[track['class_id']]
            confidence = track['confidence']
            
            # Cor baseada no ID
            color = self._get_color(track_id)
            
            # Desenhar bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"ID:{track_id} {class_name} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Background do label
            cv2.rectangle(frame, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Texto
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Desenhar trajetória
            if len(track['history']) > 1:
                points = np.array(list(track['history']), dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)
        
        return frame
    
    @staticmethod
    def _get_color(track_id: int) -> Tuple[int, int, int]:
        """Gera cor consistente para um ID"""
        np.random.seed(track_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))


if __name__ == "__main__":
    # Teste do detector
    detector = ObjectDetector()
    
    cap = cv2.VideoCapture(0)  # Webcam
    
    print("Pressione 'q' para sair")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_annotated, tracks = detector.detect_and_track(frame)
        
        cv2.imshow('Object Detection & Tracking', frame_annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()