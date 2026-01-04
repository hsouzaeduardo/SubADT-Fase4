# DOCUMENTO DE ENTREGA
# Desafio IADT - Fase 4: Análise de Vídeo com IA

## 📋 Informações do Projeto

**Título:** Sistema de Análise de Vídeo com Detecção de Anomalias  
**Disciplina:** Inteligência Artificial e Data Technology  
**Fase:** 4 (Final)  
**Data de Entrega:** 03/01/2026

## 👥 Equipe

**Integrantes:**
- Henrique Eduardo Souza

## 🔗 Links de Entrega

### Vídeo de Apresentação
**YouTube:** [[INSERIR LINK AQUI]](https://www.youtube.com/watch?v=yF0NQhjx82E)
- Duração: [15] minutos
- Conteúdo:
  - Demonstração da aplicação em funcionamento
  - Explicação das funcionalidades implementadas
  - Análise dos resultados obtidos
  - Discussão das técnicas utilizadas

### Repositório GitHub
**URL:** https://github.com/hsouzaeduardo/SubADT-Fase4/tree/feat/futebol
- Código-fonte completo
- Documentação (README.md)
- Instruções de instalação
- Exemplos de uso
- requirements.txt

## 📊 Funcionalidades Implementadas

### ✅ Requisitos Atendidos

#### 1. Detecção e Rastreamento de Objetos/Pessoas
- [x] Identificação de pessoas e objetos no vídeo
- [x] Rastreamento com IDs únicos persistentes
- [x] Registro de trajetórias de movimento
- [x] Tecnologias: YOLOv8 + Algoritmo de tracking customizado

#### 2. Classificação de Atividades
- [x] **PARADO**: Pessoa/objeto sem movimento significativo
- [x] **CAMINHANDO**: Movimento moderado (2-5 pixels/frame)
- [x] **CORRENDO**: Movimento rápido (>8 pixels/frame)
- [x] **INTERAGINDO**: Proximidade entre pessoas (<100 pixels)
- [x] **COMPORTAMENTO_ERRATICO**: Mudanças bruscas de direção
- [x] Padrões de movimentação baseados no fluxo do ambiente

#### 3. Identificação de Anomalias
- [x] **MOVIMENTO_SUBITO**: Aceleração > 10 pixels/frame² (Severidade: MÉDIA)
- [x] **VELOCIDADE_ANORMAL**: Velocidade > 8 pixels/frame (Severidade: ALTA)
- [x] **PARADA_PROLONGADA**: Imobilidade > 5 segundos (Severidade: BAIXA)
- [x] **AGLOMERACAO**: 3+ pessoas em área pequena (Severidade: MÉDIA)
- [x] **MOVIMENTO_REVERSO**: Retorno ao ponto inicial (Severidade: BAIXA)
- [x] **OBJETO_ABANDONADO**: Objeto parado > 10 segundos (Severidade: ALTA)
- [x] Categorização em 3 níveis: BAIXA, MÉDIA, ALTA

#### 4. Geração de Relatório Automático
- [x] Estatísticas detalhadas de movimentação
- [x] Alertas de eventos atípicos organizados
- [x] Insights sobre padrões de comportamento
- [x] Visualizações gráficas (gráficos de linha, pizza, timeline)
- [x] Exportação em PDF profissional

## 🎯 Tecnologias Utilizadas

### Frameworks e Bibliotecas Principais
- **YOLOv8** (Ultralytics): Detecção de objetos em tempo real
- **OpenCV**: Processamento de vídeo e visão computacional
- **NumPy/SciPy**: Cálculos numéricos e análise estatística
- **Matplotlib/Seaborn**: Visualização de dados
- **ReportLab**: Geração de relatórios PDF

### Técnicas de IA/ML Implementadas
1. **Deep Learning**: YOLOv8 para detecção de objetos
2. **Computer Vision**: Análise de movimento e trajetórias
3. **Tracking Algorithms**: Rastreamento multi-objeto com IoU
4. **Anomaly Detection**: Detecção baseada em heurísticas e thresholds
5. **Activity Recognition**: Classificação baseada em velocidade e padrões

## 📈 Resultados Obtidos

### Vídeo de Teste
**Arquivo:** [[JogoSantos.mp4]](https://youtu.be/Q1ZiaqMaQok)
**Duração:** [75] segundos
**Resolução:** [WxH]

### Estatísticas da Análise
- **Objetos detectados:** [X] únicos
- **Atividades classificadas:** [X] total
- **Anomalias detectadas:** [X] total
  - Alta severidade: [X]
  - Média severidade: [X]
  - Baixa severidade: [X]

### Performance
- **FPS de processamento:** [X] fps
- **Tempo total:** [X] minutos
- **Taxa de detecção:** [X]%

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    Entrada: Vídeo                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
           ┌──────────▼──────────┐
           │  1. DETECÇÃO        │
           │  (YOLOv8)           │
           │  - Objetos/Pessoas  │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │  2. RASTREAMENTO    │
           │  (Custom Tracker)   │
           │  - IDs únicos       │
           │  - Trajetórias      │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │  3. CLASSIFICAÇÃO   │
           │  (Activity Class.)  │
           │  - Atividades       │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │  4. ANOMALIAS       │
           │  (Anomaly Detect.)  │
           │  - Eventos atípicos │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │  5. RELATÓRIO       │
           │  (Report Gen.)      │
           │  - PDF + JSON       │
           └─────────────────────┘
```

## 📁 Estrutura de Arquivos

```
video_anomaly_detection/
├── src/
│   ├── detector.py              # Detecção e rastreamento
│   ├── activity_classifier.py   # Classificação de atividades
│   ├── anomaly_detector.py      # Detecção de anomalias
│   ├── report_generator.py      # Geração de relatórios
│   └── main.py                  # Script principal
├── data/
│   ├── input/                   # Vídeos de entrada
│   └── output/                  # Resultados
├── requirements.txt             # Dependências
├── README.md                    # Documentação principal
├── QUICKSTART.md               # Guia rápido
└── TECHNICAL_DOC.md            # Documentação técnica
```

## 🚀 Como Executar

### Instalação
```bash
git clone [LINK_DO_REPOSITORIO]
cd video_anomaly_detection
pip install -r requirements.txt
```

### Execução Básica
```bash
python src/main.py --input data/input/video.mp4
```

### Execução Completa (Recomendada)
```bash
python src/main.py \
    --input data/input/video.mp4 \
    --show-video \
    --save-video \
    --generate-report
```

## 💡 Diferenciais do Projeto

1. **Código Modular**: Arquitetura bem organizada e extensível
2. **Documentação Completa**: README, guia rápido e documentação técnica
3. **Múltiplos Níveis de Severidade**: Categorização inteligente de anomalias
4. **Relatórios Profissionais**: PDFs com visualizações e insights automáticos
5. **Performance Otimizada**: Uso eficiente de recursos computacionais
6. **Testes Automatizados**: Script de verificação do sistema

## 🎓 Aprendizados

Durante o desenvolvimento deste projeto, foram aplicados conceitos de:
- Visão Computacional
- Deep Learning para detecção de objetos
- Algoritmos de tracking
- Análise de padrões de movimento
- Detecção de anomalias
- Geração automatizada de relatórios
- Engenharia de software (modularização, documentação)

---

**Data de Submissão:** 03/1/2026
**Assinatura:** [Henrique Eduardo Souza]

## ✅ Checklist de Entrega

- [X] Vídeo gravado (máx 10 min)
- [X] Vídeo enviado para YouTube
- [X] Código no GitHub
- [X] README completo
- [X] requirements.txt
- [X] Instruções de execução
- [X] PDF com links (este documento)
- [X] Nomes na descrição do vídeo
- [X] Tema diferente do apresentado em aula
- [X] Sem relatório PDF extra (apenas vídeo)
- [X] Sem slides de apresentação (apenas vídeo)
