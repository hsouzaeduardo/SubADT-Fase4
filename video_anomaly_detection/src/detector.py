"""
Módulo de Detecção e Rastreamento de Objetos
- Tracking com IoU + gating por classe
- Sem time.time(): usa timestamp do vídeo (ou frame_number)
- Fallback simples por distância de centro para reduzir troca de IDs
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from typing import List, Dict, Tuple, Optional


class ObjectTracker:
    """Rastreador de objetos com ID único e histórico de trajetória"""

    def __init__(self, max_history: int = 30, max_age: int = 30, min_hits: int = 3):
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 1
        self.max_history = max_history
        self.max_age = max_age      # frames sem match antes de remover
        self.min_hits = min_hits

        # Thresholds do tracker
        self.iou_threshold = 0.30   # IoU mínimo para associar
        self.center_dist_threshold = 80.0  # fallback: pixels (ajuste)

    def update(self, detections: List[Dict], frame_number: int, timestamp: float) -> List[Dict]:
        """
        Atualiza tracks com novas detecções.

        Args:
            detections: Lista [{bbox, confidence, class_id, class_name?}]
            frame_number: frame atual
            timestamp: tempo do vídeo (segundos)

        Returns:
            Lista de tracks ativos
        """
        if not self.tracks:
            for det in detections:
                self._create_track(det, frame_number, timestamp)
        else:
            self._associate_detections(detections, frame_number, timestamp)

        self._cleanup_tracks(frame_number)
        return self._get_active_tracks()

    def _create_track(self, detection: Dict, frame_number: int, timestamp: float) -> int:
        track_id = self.next_id
        self.next_id += 1

        center = self._bbox_center(detection['bbox'])

        self.tracks[track_id] = {
            'id': track_id,
            'bbox': detection['bbox'],
            'class_id': detection['class_id'],
            'confidence': detection['confidence'],
            'history': deque([center], maxlen=self.max_history),

            'age': 0,                 # frames desde último match
            'hits': 1,                # quantas vezes foi matchado
            'velocity': np.array([0.0, 0.0]),

            'first_seen_frame': frame_number,
            'last_seen_frame': frame_number,
            'first_seen_ts': timestamp,
            'last_seen_ts': timestamp,
        }

        return track_id

    def _associate_detections(self, detections: List[Dict], frame_number: int, timestamp: float):
        # Se não há detecções, envelhece todos
        if not detections:
            for t in self.tracks.values():
                t['age'] += 1
            return

        track_ids = list(self.tracks.keys())

        # Custos: 1 - IoU (apenas quando class_id bate; senão custo alto)
        cost_matrix = np.ones((len(detections), len(track_ids)), dtype=np.float32) * 1e6

        for i, det in enumerate(detections):
            for j, tid in enumerate(track_ids):
                tr = self.tracks[tid]

                # gating por classe
                if det['class_id'] != tr['class_id']:
                    continue

                iou = self._calculate_iou(det['bbox'], tr['bbox'])
                if iou > 0:
                    cost_matrix[i, j] = 1.0 - iou

        # Greedy por menor custo (mantém seu estilo)
        matched_tracks = set()
        matched_detections = set()

        flat = [(cost_matrix[i, j], i, j) for i in range(len(detections)) for j in range(len(track_ids))]
        flat.sort(key=lambda x: x[0])

        for cost, det_idx, track_idx in flat:
            if det_idx in matched_detections or track_idx in matched_tracks:
                continue

            # Se custo é "infinito", não há match permitido
            if cost >= 1e5:
                continue

            # IoU mínimo: cost < (1 - iou_threshold)
            if cost < (1.0 - self.iou_threshold):
                tid = track_ids[track_idx]
                self._update_track(tid, detections[det_idx], frame_number, timestamp)
                matched_tracks.add(track_idx)
                matched_detections.add(det_idx)

        # Fallback por distância de centro (para casos com IoU ruim)
        # Só tenta para detecções não associadas
        for i, det in enumerate(detections):
            if i in matched_detections:
                continue

            det_center = np.array(self._bbox_center(det['bbox']))
            best_tid = None
            best_dist = float('inf')

            for j, tid in enumerate(track_ids):
                if j in matched_tracks:
                    continue

                tr = self.tracks[tid]
                if det['class_id'] != tr['class_id']:
                    continue

                tr_center = np.array(self._bbox_center(tr['bbox']))
                dist = float(np.linalg.norm(det_center - tr_center))

                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None and best_dist <= self.center_dist_threshold:
                self._update_track(best_tid, det, frame_number, timestamp)
                matched_detections.add(i)
                # marca track como matchado (encontrar índice)
                matched_tracks.add(track_ids.index(best_tid))

        # Cria novos tracks para detecções não associadas
        for i, det in enumerate(detections):
            if i not in matched_detections:
                self._create_track(det, frame_number, timestamp)

        # Envelhece tracks não associados
        for j, tid in enumerate(track_ids):
            if j not in matched_tracks and tid in self.tracks:
                self.tracks[tid]['age'] += 1

    def _update_track(self, track_id: int, detection: Dict, frame_number: int, timestamp: float):
        tr = self.tracks[track_id]

        old_center = np.array(self._bbox_center(tr['bbox']))
        new_center = np.array(self._bbox_center(detection['bbox']))
        velocity = new_center - old_center  # pixels/frame (se quiser m/s, converta depois)

        tr['bbox'] = detection['bbox']
        tr['confidence'] = detection['confidence']
        tr['history'].append(tuple(new_center))
        tr['age'] = 0
        tr['hits'] += 1
        tr['velocity'] = velocity

        tr['last_seen_frame'] = frame_number
        tr['last_seen_ts'] = timestamp

    def _cleanup_tracks(self, frame_number: int):
        to_remove = []
        for tid, tr in self.tracks.items():
            if tr['age'] > self.max_age:
                to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

    def _get_active_tracks(self) -> List[Dict]:
        active = []
        for tr in self.tracks.values():
            if tr['hits'] >= self.min_hits:
                active.append(tr)
        return active

    @staticmethod
    def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _calculate_iou(b1: List[float], b2: List[float]) -> float:
        x1_1, y1_1, x2_1, y2_1 = b1
        x1_2, y1_2, x2_2, y2_2 = b2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        inter = (x2_i - x1_i) * (y2_i - y1_i)
        a1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        a2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = a1 + a2 - inter
        return float(inter / union) if union > 0 else 0.0


class ObjectDetector:
    """Detector de objetos usando YOLOv8 + tracker"""

    def __init__(self, model_name: str = 'yolov8n.pt', confidence: float = 0.5):
        self.model = YOLO(model_name)
        self.confidence = confidence

        # Para futebol: pessoa + bola (COCO "sports ball" = 32)
        # Se seu modelo não detecta bola bem, isso vai precisar de modelo específico.
        self.target_classes = [0, 32]
        self.class_names = self.model.names

        self.tracker = ObjectTracker(max_history=45, max_age=30, min_hits=2)

    def detect_and_track(self, frame: np.ndarray, frame_number: int, timestamp: float) -> Tuple[np.ndarray, List[Dict]]:
        results = self.model(frame, conf=self.confidence, classes=self.target_classes, verbose=False)

        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': self.class_names.get(cls, str(cls)) if isinstance(self.class_names, dict) else self.class_names[cls]
                })

        tracks = self.tracker.update(detections, frame_number=frame_number, timestamp=timestamp)
        frame_annotated = self._annotate_frame(frame.copy(), tracks)
        return frame_annotated, tracks

    def _annotate_frame(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        for tr in tracks:
            x1, y1, x2, y2 = map(int, tr['bbox'])
            tid = tr['id']
            cls = tr['class_id']
            conf = tr['confidence']

            name = self.class_names.get(cls, str(cls)) if isinstance(self.class_names, dict) else self.class_names[cls]
            color = self._get_color(tid)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{tid} {name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if len(tr['history']) > 1:
                pts = np.array(list(tr['history']), dtype=np.int32)
                cv2.polylines(frame, [pts], False, color, 2)

        return frame

    @staticmethod
    def _get_color(track_id: int) -> Tuple[int, int, int]:
        np.random.seed(track_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
