# Documentação Técnica - Sistema de Análise de Vídeo

## Arquitetura do Sistema

### Visão Geral

O sistema é composto por 4 módulos principais que trabalham em pipeline:

```
Vídeo → [Detecção] → [Classificação] → [Anomalias] → [Relatório]
```

### 1. Módulo de Detecção e Rastreamento (`detector.py`)

#### ObjectDetector
Responsável pela detecção de objetos usando YOLOv8.

**Características:**
- Modelo: YOLOv8 (Nano por padrão para performance)
- Classes detectadas: pessoas, mochilas, bolsas, malas, garrafas, xícaras, celulares
- Confiança configurável (padrão: 0.5)

**Fluxo:**
1. Frame → YOLOv8 → Detecções (bbox, classe, confiança)
2. Detecções → ObjectTracker → Tracks com IDs únicos

#### ObjectTracker
Implementa rastreamento multi-objeto customizado.

**Algoritmo:**
- Associação de detecções usando IoU (Intersection over Union)
- Kalman filter implícito através de histórico de velocidades
- Manutenção de trajetórias (histórico de posições)

**Parâmetros importantes:**
```python
max_history = 30        # Frames de histórico
max_age = 30           # Frames sem detecção antes de remover
iou_threshold = 0.7    # Threshold para associação
min_hits = 3           # Mínimo de detecções para track válido
```

### 2. Módulo de Classificação de Atividades (`activity_classifier.py`)

#### ActivityClassifier
Classifica atividades baseado em análise de movimento.

**Atividades Detectadas:**

| Atividade | Critério | Threshold |
|-----------|----------|-----------|
| PARADO | Velocidade < 2.0 | pixels/frame |
| CAMINHANDO | 2.0 < Velocidade < 5.0 | pixels/frame |
| CORRENDO | Velocidade > 8.0 | pixels/frame |
| INTERAGINDO | Distância < 100 | pixels entre pessoas |
| COMPORTAMENTO_ERRATICO | Mudanças de direção > 45° | em 3+ ocorrências |

**Algoritmo de Classificação:**

```python
def classify(track):
    # 1. Calcular velocidade
    velocity = track.velocity
    speed = norm(velocity)
    
    # 2. Verificar comportamento errático
    if is_erratic(track):
        return 'COMPORTAMENTO_ERRATICO'
    
    # 3. Classificar por velocidade
    if speed < STOPPED_THRESHOLD:
        return 'PARADO'
    elif speed < WALKING_THRESHOLD:
        return 'CAMINHANDO'
    else:
        return 'CORRENDO'
    
    # 4. Verificar interações
    if check_proximity(track, other_tracks):
        return 'INTERAGINDO'
```

**Detecção de Comportamento Errático:**
- Janela deslizante de 5 frames
- Calcula direção de movimento em cada janela
- Compara diferenças angulares
- Se > 3 mudanças > 45°, considera errático

### 3. Módulo de Detecção de Anomalias (`anomaly_detector.py`)

#### AnomalyDetector
Identifica eventos atípicos usando múltiplas heurísticas.

**Anomalias Detectadas:**

| Tipo | Descrição | Severidade | Critério |
|------|-----------|------------|----------|
| MOVIMENTO_SUBITO | Aceleração repentina | MEDIA | accel > 10 px/frame² |
| VELOCIDADE_ANORMAL | Velocidade excessiva | ALTA | speed > 8 px/frame |
| PARADA_PROLONGADA | Imobilidade prolongada | BAIXA | stopped > 5s |
| AGLOMERACAO | Múltiplas pessoas próximas | MEDIA | 3+ pessoas < 80px |
| MOVIMENTO_REVERSO | Retorno ao ponto inicial | BAIXA | dist < 50px do início |
| OBJETO_ABANDONADO | Objeto estático prolongado | ALTA | objeto parado > 10s |

**Algoritmo de Detecção:**

```python
def detect(tracks, activities, frame, timestamp):
    anomalies = []
    
    for track in tracks:
        # Verificar múltiplas condições
        if check_sudden_movement(track):
            anomalies.append(create_anomaly('MOVIMENTO_SUBITO'))
        
        if check_high_speed(track):
            anomalies.append(create_anomaly('VELOCIDADE_ANORMAL'))
        
        if check_prolonged_stop(track, activity):
            anomalies.append(create_anomaly('PARADA_PROLONGADA'))
        
        # ... outras verificações
    
    # Verificações coletivas
    if crowding := check_crowding(tracks):
        anomalies.extend(crowding)
    
    return anomalies
```

**Níveis de Severidade:**
- 🔴 **ALTA**: Requer atenção imediata (velocidade anormal, objeto abandonado)
- 🟡 **MEDIA**: Comportamentos suspeitos (movimento súbito, aglomeração)
- 🟢 **BAIXA**: Eventos incomuns mas não críticos (parada prolongada, movimento reverso)

### 4. Módulo de Geração de Relatórios (`report_generator.py`)

#### ReportGenerator
Cria relatórios em PDF com análises visuais.

**Componentes do Relatório:**

1. **Informações do Vídeo**
   - Nome, duração, FPS, total de frames
   
2. **Resumo Executivo**
   - Total de objetos rastreados
   - Atividades detectadas
   - Anomalias por severidade
   
3. **Estatísticas de Detecção**
   - Gráfico de linha: objetos ao longo do tempo
   
4. **Classificação de Atividades**
   - Tabela com contagens
   - Gráfico de pizza: distribuição
   
5. **Detecção de Anomalias**
   - Tabela por severidade
   - Tabela por tipo
   - Timeline de anomalias
   
6. **Conclusões e Insights**
   - Análise automatizada dos padrões detectados

**Visualizações:**
- Matplotlib/Seaborn para gráficos
- ReportLab para geração de PDF
- Esquema de cores consistente

## Pipeline de Processamento

### Fluxo Completo

```python
# Inicialização
detector = ObjectDetector(confidence=0.5)
classifier = ActivityClassifier(frame_rate=fps)
anomaly_detector = AnomalyDetector(frame_rate=fps)

for frame_number, frame in video:
    # 1. Detecção e Rastreamento
    frame_annotated, tracks = detector.detect_and_track(frame)
    # → Retorna: lista de tracks com ID, bbox, velocidade
    
    # 2. Classificação de Atividades
    activities = classifier.classify(tracks, frame_number)
    # → Retorna: {track_id: atividade}
    
    # 3. Detecção de Anomalias
    timestamp = frame_number / fps
    anomalies = anomaly_detector.detect(
        tracks, activities, frame_number, timestamp
    )
    # → Retorna: lista de anomalias detectadas
    
    # 4. Anotação do Frame
    frame_final = annotate_frame(
        frame_annotated, tracks, activities, anomalies
    )
    
    # 5. Salvar dados
    save_frame_data(frame_number, tracks, activities, anomalies)

# 6. Gerar Relatório
report_gen = ReportGenerator(output_dir)
pdf_path = report_gen.generate_report(video_path, analysis_data)
```

## Estrutura de Dados

### Track Object
```python
{
    'id': int,                    # ID único do track
    'bbox': [x1, y1, x2, y2],    # Bounding box
    'class_id': int,              # ID da classe COCO
    'class_name': str,            # Nome da classe
    'confidence': float,          # Confiança da detecção
    'velocity': np.array([vx, vy]),  # Vetor velocidade
    'history': deque([...]),      # Histórico de posições
    'age': int,                   # Frames desde última detecção
    'hits': int,                  # Total de detecções
    'last_seen': float            # Timestamp
}
```

### Anomaly Object
```python
{
    'type': str,                  # Tipo da anomalia
    'description': str,           # Descrição
    'severity': str,              # 'BAIXA', 'MEDIA', 'ALTA'
    'frame': int,                 # Frame onde ocorreu
    'timestamp': float,           # Tempo em segundos
    'track_id': int,              # ID do track (se aplicável)
    'location': (x, y),           # Posição no frame
    'bbox': [x1, y1, x2, y2]     # Bounding box (se aplicável)
}
```

### Analysis Data
```python
{
    'video_path': str,
    'fps': float,
    'duration': float,
    'total_frames': int,
    'frames_data': [              # Dados de cada frame
        {
            'frame': int,
            'tracks_count': int,
            'tracks': [...],
            'anomalies': [...]
        }
    ],
    'detection_stats': {
        'frames': [int, ...],
        'object_counts': [int, ...]
    },
    'activity_stats': {
        'total_tracks': int,
        'activity_counts': {...}
    },
    'anomaly_stats': {
        'total_anomalies': int,
        'by_type': {...},
        'by_severity': {...},
        'timeline': [...]
    },
    'summary': {
        'total_tracks': int,
        'total_activities': int,
        'total_anomalies': int,
        'high_severity_anomalies': int
    }
}
```

## Performance e Otimizações

### Benchmarks (estimados)

| Configuração | FPS Processamento | Uso de Memória |
|--------------|-------------------|----------------|
| YOLOv8n + CPU | 10-15 fps | ~2GB |
| YOLOv8n + GPU | 50-80 fps | ~4GB |
| YOLOv8m + CPU | 3-5 fps | ~3GB |
| YOLOv8m + GPU | 30-50 fps | ~6GB |

### Dicas de Otimização

1. **Para processamento mais rápido:**
   - Use YOLOv8n (nano)
   - Reduza resolução do vídeo
   - Aumente threshold de confiança
   - Use GPU

2. **Para maior precisão:**
   - Use YOLOv8m ou YOLOv8l
   - Reduza threshold de confiança
   - Aumente histórico de tracking

3. **Para economizar memória:**
   - Reduza max_history nos trackers
   - Processe em lotes
   - Limpe dados antigos regularmente

## Limitações Conhecidas

1. **Oclusão:** Objetos totalmente ocultos perdem tracking
2. **Iluminação:** Mudanças drásticas afetam detecção
3. **Câmera móvel:** Sistema otimizado para câmera fixa
4. **Escala:** Objetos muito pequenos podem não ser detectados
5. **Velocidade:** Movimentos extremamente rápidos podem perder frames

## Extensões Possíveis

1. **Re-identificação:** Reconhecer mesmo objeto após oclusão
2. **Zonas de interesse:** Definir áreas específicas para monitorar
3. **Regras customizadas:** Definir regras de negócio específicas
4. **Multi-câmera:** Rastreamento entre múltiplas câmeras
5. **Deep Learning para atividades:** Usar redes neurais para classificação

## Referências

- YOLOv8: https://github.com/ultralytics/ultralytics
- DeepSORT: https://arxiv.org/abs/1703.07402
- Anomaly Detection in Videos: https://arxiv.org/abs/1801.04264
- Activity Recognition: https://arxiv.org/abs/1705.07750