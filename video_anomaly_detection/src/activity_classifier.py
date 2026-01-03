import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from collections import deque
from scipy.spatial.distance import euclidean


class SoccerActivityClassifier:
    """
    Classificador de atividades para futebol (movimento + eventos básicos).
    Reaproveita a lógica do ActivityClassifier e adiciona bola/posse/eventos.
    """

    # Ajuste conforme seu detector (COCO "sports ball" costuma ser 32)
    BALL_CLASS_ID = 32

    ACTIVITIES = {
        # Movimento
        'PARADO': 'Sem movimento significativo',
        'CAMINHANDO': 'Baixa velocidade',
        'TROTANDO': 'Velocidade moderada',
        'CORRENDO': 'Alta velocidade',
        'SPRINT': 'Velocidade muito alta',
        'COMPORTAMENTO_ERRATICO': 'Mudanças bruscas de direção',
        # Futebol (contexto)
        'POSSE_DE_BOLA': 'Jogador em controle/posse da bola',
        'DISPUTA': 'Dois (ou mais) jogadores muito próximos da bola',
        'PASSE': 'Bola transita de um jogador para outro',
        'CHUTE': 'Ação de chute (pico de velocidade/aceleração da bola)'
    }

    def __init__(
        self,
        frame_rate: float = 30.0,
        pixel_to_world_fn: Optional[Callable[[Tuple[float, float]], Tuple[float, float]]] = None
    ):
        """
        Args:
            frame_rate: fps do vídeo
            pixel_to_world_fn: função que converte (x,y) em pixels para (X,Y) em metros no campo.
                               Se None, usa pixels como unidade (menos robusto).
        """
        self.frame_rate = frame_rate
        self.pixel_to_world_fn = pixel_to_world_fn

        self.history_size = 45  # futebol se beneficia de janela maior
        self.position_history = {}   # {track_id: deque[(x,y)]}
        self.activity_history = {}   # {track_id: deque[str]}

        # Bola
        self.ball_history = deque(maxlen=self.history_size)  # deque[(x,y)]
        self.ball_speed_history = deque(maxlen=self.history_size)  # deque[float]

        # Posse
        self.current_possession: Optional[int] = None
        self.possession_streak = 0

        # Thresholds em m/s (se pixel_to_world_fn existir); caso contrário, viram "unidades/s"
        self.thresholds = {
            # Movimento (m/s)
            'stopped_speed': 0.3,
            'walking_speed': 2.0,
            'jogging_speed': 4.0,
            'running_speed': 7.0,

            # Interações (metros)
            'possession_distance': 1.5,    # jogador-bola
            'dispute_distance': 2.0,       # mais de 1 jogador próximo da bola
            'interaction_distance': 1.5,   # jogador-jogador (pressão)

            # Errático
            'direction_change': 45.0,
            'erratic_changes': 3,

            # Eventos bola
            'pass_ball_speed': 6.0,        # m/s (ajuste)
            'shot_ball_speed': 12.0,       # m/s (ajuste)
            'possession_min_frames': 6     # evita flicker de posse (ex.: 0.2s em 30fps)
        }

    def classify(self, tracks: List[Dict], frame_number: int) -> Dict[int, str]:
        """
        Retorna {track_id: atividade}. Também detecta eventos (passe/chute) e marca
        o jogador envolvido quando possível.
        """
        activities: Dict[int, str] = {}

        # Separar bola e pessoas
        players = [t for t in tracks if t.get('class_id') == 0]
        balls = [t for t in tracks if t.get('class_id') == self.BALL_CLASS_ID]

        # Atualizar históricos
        self._update_position_history(players)
        self._update_ball_history(balls)

        # Detectar posse/disputa com base na bola (se existir)
        possession_info = self._infer_possession(players)

        # Classificar jogadores
        for p in players:
            pid = p['id']

            # Base (movimento/errático)
            activity = self._classify_movement(pid, p)

            # Contexto futebol
            if possession_info is not None:
                possessor_id, in_dispute = possession_info

                if in_dispute and self._is_player_near_ball(pid):
                    activity = 'DISPUTA'
                elif possessor_id == pid:
                    activity = 'POSSE_DE_BOLA'

            activities[pid] = activity
            self._push_activity(pid, activity)

        # Detectar eventos de bola (passe/chute)
        event = self._detect_ball_event()
        if event and self.current_possession is not None:
            # Atribui evento ao possuidor (heurística simples)
            activities[self.current_possession] = event
            self._push_activity(self.current_possession, event)

        return activities

    # ----------------------------
    # Núcleo: movimento / errático
    # ----------------------------
    def _classify_movement(self, track_id: int, track: Dict) -> str:
        # se você já calcula velocity externo e passa em m/s, pode usar direto
        speed = self._estimate_speed(track_id, track)

        if self._is_erratic_behavior(track_id):
            return 'COMPORTAMENTO_ERRATICO'

        if speed < self.thresholds['stopped_speed']:
            return 'PARADO'
        if speed < self.thresholds['walking_speed']:
            return 'CAMINHANDO'
        if speed < self.thresholds['jogging_speed']:
            return 'TROTANDO'
        if speed < self.thresholds['running_speed']:
            return 'CORRENDO'
        return 'SPRINT'

    def _estimate_speed(self, track_id: int, track: Dict) -> float:
        """
        Estima velocidade com base no histórico de posições (preferível),
        porque nem sempre 'velocity' vem consistente do tracker.
        """
        if track_id not in self.position_history:
            return 0.0

        hist = self.position_history[track_id]
        if len(hist) < 2:
            return 0.0

        p1 = np.array(hist[-2])
        p2 = np.array(hist[-1])
        dist = np.linalg.norm(p2 - p1)  # em metros se já convertidos, senão pixels
        return dist * self.frame_rate   # unidade/segundo

    def _is_erratic_behavior(self, track_id: int) -> bool:
        if track_id not in self.position_history:
            return False
        positions = list(self.position_history[track_id])
        if len(positions) < 10:
            return False

        direction_changes = 0
        window_size = 5

        for i in range(len(positions) - window_size * 2):
            dir1 = self._calculate_direction(positions[i:i + window_size])
            dir2 = self._calculate_direction(positions[i + window_size:i + window_size * 2])

            if dir1 is not None and dir2 is not None:
                angle_diff = abs(self._angle_difference(dir1, dir2))
                if angle_diff > self.thresholds['direction_change']:
                    direction_changes += 1

        return direction_changes >= self.thresholds['erratic_changes']

    # ----------------------------
    # Bola / posse / eventos
    # ----------------------------
    def _update_ball_history(self, balls: List[Dict]):
        if not balls:
            # sem bola detectada no frame: não atualiza
            return

        # Se houver mais de uma detecção, pega a de maior confiança se existir
        ball = max(balls, key=lambda b: b.get('confidence', 0.0))
        center_px = self._get_center(ball['bbox'])
        center = self._to_world(center_px)

        # velocidade bola
        if self.ball_history:
            prev = np.array(self.ball_history[-1])
            curr = np.array(center)
            dist = np.linalg.norm(curr - prev)
            speed = dist * self.frame_rate
        else:
            speed = 0.0

        self.ball_history.append(center)
        self.ball_speed_history.append(speed)

    def _infer_possession(self, players: List[Dict]) -> Optional[Tuple[Optional[int], bool]]:
        """
        Retorna (possessor_id, in_dispute)
        - possessor_id pode ser None
        - in_dispute True quando 2+ jogadores estão muito perto da bola
        """
        if not self.ball_history:
            self.current_possession = None
            self.possession_streak = 0
            return None

        ball = self.ball_history[-1]

        # distâncias jogador-bola
        dists = []
        for p in players:
            pid = p['id']
            center = self._get_center(p['bbox'])
            center = self._to_world(center)
            d = euclidean(center, ball)
            dists.append((d, pid))

        if not dists:
            return (None, False)

        dists.sort(key=lambda x: x[0])
        closest_d, closest_id = dists[0]

        # disputa: 2 jogadores dentro de dispute_distance
        near_ids = [pid for d, pid in dists if d < self.thresholds['dispute_distance']]
        in_dispute = (len(near_ids) >= 2)

        # posse: mais próximo dentro de possession_distance por N frames
        if closest_d < self.thresholds['possession_distance']:
            if self.current_possession == closest_id:
                self.possession_streak += 1
            else:
                self.current_possession = closest_id
                self.possession_streak = 1
        else:
            self.current_possession = None
            self.possession_streak = 0

        possessor = self.current_possession if self.possession_streak >= self.thresholds['possession_min_frames'] else None
        return (possessor, in_dispute)

    def _is_player_near_ball(self, player_id: int) -> bool:
        if not self.ball_history or player_id not in self.position_history:
            return False
        ball = self.ball_history[-1]
        player_pos = self.position_history[player_id][-1]
        return euclidean(player_pos, ball) < self.thresholds['dispute_distance']

    def _detect_ball_event(self) -> Optional[str]:
        """
        Detecta eventos simples usando velocidade da bola:
        - CHUTE: pico alto e persistente
        - PASSE: velocidade moderada e transição de posse
        """
        if len(self.ball_speed_history) < 6:
            return None

        recent = list(self.ball_speed_history)[-6:]
        v_now = recent[-1]
        v_avg = sum(recent) / len(recent)

        # Heurística de "pico": agora bem acima da média recente
        spike = v_now > (v_avg * 1.6) and v_now > self.thresholds['pass_ball_speed']

        if v_now >= self.thresholds['shot_ball_speed'] and spike:
            return 'CHUTE'

        if v_now >= self.thresholds['pass_ball_speed'] and spike:
            return 'PASSE'

        return None

    # ----------------------------
    # Infra: históricos e utilitários
    # ----------------------------
    def _update_position_history(self, tracks: List[Dict]):
        for track in tracks:
            track_id = track['id']
            center_px = self._get_center(track['bbox'])
            center = self._to_world(center_px)

            if track_id not in self.position_history:
                self.position_history[track_id] = deque(maxlen=self.history_size)
            self.position_history[track_id].append(center)

    def _push_activity(self, track_id: int, activity: str):
        if track_id not in self.activity_history:
            self.activity_history[track_id] = deque(maxlen=self.history_size)
        self.activity_history[track_id].append(activity)

    def _to_world(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        if self.pixel_to_world_fn is None:
            return pt
        return self.pixel_to_world_fn(pt)

    def get_activity_statistics(self) -> Dict:
      """Retorna estatísticas de atividades ao longo do tempo."""
      stats = {
          'total_tracks': len(self.activity_history),
          'activity_counts': {},
          'track_activities': {}
      }

      # Inicializa contadores para todas as atividades conhecidas
      for activity in self.ACTIVITIES.keys():
          stats['activity_counts'][activity] = 0

      # Processa histórico por track
      for track_id, activities in self.activity_history.items():
          activity_list = list(activities)

          if not activity_list:
              continue

          # Atividade mais frequente (modo)
          most_common = max(set(activity_list), key=activity_list.count)

          # Distribuição por track
          stats['track_activities'][track_id] = {
              'most_common': most_common,
              'distribution': {
                  act: activity_list.count(act) / len(activity_list)
                  for act in set(activity_list)
              }
          }

          # Contabiliza no agregado
          for act in activity_list:
              # Se surgir uma atividade nova (ex.: você adicionou depois), não quebra
              if act not in stats['activity_counts']:
                  stats['activity_counts'][act] = 0
              stats['activity_counts'][act] += 1

      return stats

    def get_activity_description(self, activity: str) -> str:
        """Retorna descrição de uma atividade."""
        return self.ACTIVITIES.get(activity, "Atividade desconhecida")



    @staticmethod
    def _get_center(bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _calculate_direction(positions: List[Tuple[float, float]]) -> Optional[float]:
        if len(positions) < 2:
            return None
        start = np.array(positions[0])
        end = np.array(positions[-1])
        displacement = end - start
        if np.linalg.norm(displacement) < 1e-6:
            return None
        return float(np.degrees(np.arctan2(displacement[1], displacement[0])))

    @staticmethod
    def _angle_difference(angle1: float, angle2: float) -> float:
        diff = abs(angle1 - angle2)
        return diff if diff <= 180 else 360 - diff
