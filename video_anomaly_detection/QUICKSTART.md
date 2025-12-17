# Guia Rápido de Início - Sistema de Análise de Vídeo

## 🚀 Instalação Rápida

### 1. Preparar ambiente

```bash
# Clonar o repositório
git clone <seu-repositorio>
cd video_anomaly_detection

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Preparar diretórios

```bash
mkdir -p data/input data/output models
```

### 3. Baixar vídeo de teste

Coloque seu vídeo em `data/input/` ou use o vídeo fornecido:
https://drive.google.com/file/d/1L4pZEPDW3P4GZfYZwyszgoDIpLiiEMfU/view?usp=sharing

## 📹 Uso Básico

### Análise simples (mais rápida)

```bash
python src/main.py --input data/input/seu_video.mp4
```

### Análise completa (com todas as features)

```bash
python src/main.py \
    --input data/input/seu_video.mp4 \
    --output data/output \
    --show-video \
    --save-video \
    --generate-report
```

### Análise com confiança ajustada

```bash
python src/main.py \
    --input data/input/seu_video.mp4 \
    --confidence 0.7 \
    --save-video \
    --generate-report
```

## 🎯 Casos de Uso Comuns

### 1. Análise rápida sem salvar nada
```bash
python src/main.py --input video.mp4
```

### 2. Análise com visualização em tempo real
```bash
python src/main.py --input video.mp4 --show-video
```

### 3. Gerar apenas o relatório
```bash
python src/main.py --input video.mp4 --generate-report
```

### 4. Salvar vídeo processado
```bash
python src/main.py --input video.mp4 --save-video
```

### 5. Análise completa (recomendado para apresentação)
```bash
python src/main.py \
    --input video.mp4 \
    --show-video \
    --save-video \
    --generate-report
```

## 📊 Saídas Geradas

Após a análise, você encontrará em `data/output/`:

- `processed_video.mp4` - Vídeo com anotações (se `--save-video`)
- `relatorio_YYYYMMDD_HHMMSS.pdf` - Relatório completo (se `--generate-report`)
- `analysis_data.json` - Dados brutos da análise
- Gráficos auxiliares (`.png`)

## 🔧 Troubleshooting

### Erro: "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### Erro: "CUDA not available"
O sistema funcionará em CPU. Para usar GPU:
1. Instale CUDA Toolkit
2. Reinstale PyTorch com suporte CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Vídeo muito lento para processar
Reduza a resolução ou use um modelo mais leve:
```python
# Edite src/detector.py, linha de inicialização do modelo:
self.model = YOLO('yolov8n.pt')  # n = nano (mais rápido)
# Opções: yolov8n < yolov8s < yolov8m < yolov8l < yolov8x
```

### Memória insuficiente
Processo o vídeo em lotes menores ou reduza a resolução de entrada.

## 📝 Exemplos Interativos

Execute o script de exemplos:

```bash
python examples.py
```

Ou teste diretamente com um vídeo:

```bash
python examples.py data/input/seu_video.mp4
```

## 🎥 Preparando para Apresentação

1. **Processe o vídeo fornecido**:
```bash
python src/main.py \
    --input data/input/video_desafio.mp4 \
    --show-video \
    --save-video \
    --generate-report \
    --output resultados_finais
```

2. **Grave a apresentação** mostrando:
   - Execução do comando
   - Vídeo sendo processado em tempo real
   - Relatório PDF gerado
   - Estatísticas finais no terminal

3. **Organize os arquivos**:
   - Vídeo processado
   - Relatório PDF
   - Screenshots das detecções
   - Código no GitHub

## 💡 Dicas

- Use `--confidence 0.6` ou `0.7` para ambientes com muito ruído
- `--show-video` desacelera o processamento mas é ótimo para demonstrações
- Gere o relatório sempre - ele contém insights valiosos
- Teste primeiro com vídeos curtos (30-60 segundos)

## 🔗 Links Úteis

- Documentação YOLOv8: https://docs.ultralytics.com/
- OpenCV Docs: https://docs.opencv.org/
- Issues e Dúvidas: Discord do curso

## ✅ Checklist de Entrega

- [ ] Código no GitHub com README completo
- [ ] requirements.txt com todas as dependências
- [ ] Vídeo de apresentação (máx 10 min)
- [ ] Link do vídeo no YouTube
- [ ] PDF com link do GitHub e vídeo
- [ ] Código testado e funcionando
- [ ] Comentários explicativos no código
- [ ] Instruções claras de execução