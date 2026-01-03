# Sistema de Análise de Vídeo com Detecção de Anomalias

## 📋 Descrição do Projeto

Sistema avançado de análise de vídeo que utiliza técnicas de visão computacional e inteligência artificial para:
- Detectar e rastrear objetos e pessoas
- Classificar atividades em tempo real
- Identificar comportamentos anômalos
- Gerar relatórios automáticos com insights

## 🎯 Funcionalidades Implementadas

### 1. Detecção e Rastreamento
- **YOLOv8**: Detecção de objetos em tempo real
- **DeepSORT**: Rastreamento multi-objeto com IDs únicos
- **Tracking persistente**: Mantém identidade dos objetos ao longo do vídeo

### 2. Classificação de Atividades
Atividades detectadas:
- **Caminhando**: Movimento moderado em direção definida
- **Correndo**: Movimento rápido (velocidade > 5 pixels/frame)
- **Parado**: Sem movimento significativo (< 2 pixels/frame)
- **Interagindo**: Proximidade entre pessoas (< 100 pixels)
- **Comportamento errático**: Mudanças bruscas de direção

### 3. Detecção de Anomalias
Eventos anômalos identificados:
- **Movimento súbito**: Aceleração > 10 pixels/frame²
- **Velocidade anormal**: Velocidade > 8 pixels/frame
- **Parada prolongada**: Imobilidade > 5 segundos
- **Aglomeração**: Mais de 3 pessoas em área pequena
- **Movimento reverso**: Retorno ao ponto de origem

Severidade das anomalias:
- 🟢 **BAIXA**: Comportamentos levemente atípicos
- 🟡 **MÉDIA**: Padrões que requerem atenção
- 🔴 **ALTA**: Eventos críticos que necessitam intervenção

### 4. Relatórios Automáticos
- Estatísticas de movimentação
- Gráficos de atividades ao longo do tempo
- Lista de anomalias detectadas
- Heatmap de movimentação
- Exportação em PDF

## 🚀 Instalação

### Requisitos
- Python 3.8+
- CUDA (opcional, para GPU)

### Instalação das dependências

```bash
# Clone o repositório
git clone <seu-repositorio>
cd video_anomaly_detection

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 1. Atualizar pip, setuptools e wheel
python -m pip install --upgrade pip setuptools wheel
# Instale as dependências
pip install -r requirements.txt
```

## 📁 Estrutura do Projeto

```
video_anomaly_detection/
├── src/
│   ├── detector.py           # Detecção e rastreamento de objetos
│   ├── activity_classifier.py # Classificação de atividades
│   ├── anomaly_detector.py   # Detecção de anomalias
│   ├── report_generator.py   # Geração de relatórios
│   └── main.py               # Script principal
├── data/
│   ├── input/                # Vídeos de entrada
│   └── output/               # Resultados processados
├── models/                   # Modelos treinados
├── requirements.txt          # Dependências
└── README.md                # Este arquivo
```

## 💻 Como Usar

### Processamento de vídeo

```bash
python src/main.py --input data/input/Hack.mp4 --output data/output/
```

### Parâmetros disponíveis

```bash
--input PATH          # Caminho do vídeo de entrada (obrigatório)
--output PATH         # Diretório de saída (padrão: data/output/)
--confidence FLOAT    # Confiança mínima para detecção (padrão: 0.5)
--show-video         # Exibir processamento em tempo real
--save-video         # Salvar vídeo processado
--generate-report    # Gerar relatório PDF
```

### Exemplo completo

```bash
python src/main.py \
    --input data/input/surveillance.mp4 \
    --output results/ \
    --confidence 0.6 \
    --show-video \
    --save-video \
    --generate-report
```

## 📊 Saídas Geradas

1. **Vídeo processado**: Com bounding boxes, IDs e classificações
2. **Dados JSON**: Informações detalhadas de cada frame
3. **Relatório PDF**: Análise completa com gráficos e estatísticas
4. **Logs**: Registro de todos os eventos detectados

## 🔧 Configuração Avançada

### Ajuste de sensibilidade de anomalias

Edite `src/anomaly_detector.py`:

```python
self.thresholds = {
    'sudden_movement': 10,  # Pixels/frame²
    'high_speed': 8,        # Pixels/frame
    'stopped_duration': 5,  # Segundos
}
```

### Personalização de atividades

Edite `src/activity_classifier.py` para adicionar novas atividades.

## 🎥 Demonstração

Vídeo de demonstração: [Link do YouTube]

## 👥 Equipe

- Henrique Eduardo Souza

## 📝 Licença

MIT License

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue ou pull request.

## 📧 Contato

Para dúvidas ou sugestões, entre em contato através do Discord do curso.