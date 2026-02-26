# Melhorias Aplicadas - ALPR Camera

## 🐛 Problema Identificado
O modelo YOLO original era **yolov8n** (Nano - muito pequeno), que não detecta bem veículos. Além disso, o pipeline de OCR era muito simples.

## ✅ Soluções Aplicadas

### 1. **Modelo de Detecção Melhorado**
- ❌ Antes: `yolov8n.pt` (muito pequeno)
- ✅ Depois: `yolov8m.pt` (melhor precisão)

Alternativa para melhor qualidade:
```bash
# No .env, altere para:
PURE_YOLO_MODEL_PATH="../ai/resources/yolov8l.pt"  # Ainda melhor, mais lento
```

### 2. **Pré-processamento de Placa Otimizado**
- ✅ **CLAHE**: Aumentado de `clipLimit=2.0` para `3.0`, tileGridSize `10x10`
  - Aplicado no canal V (HSV) para melhor resultado em placas coloridas
  
- ✅ **Warp Plate**: Aumentado de 300x100 para **400x130 pixels**
  - Melhor resolução para OCR
  
- ✅ **Enhance Plate**: Ajustes mais agressivos
  - Contraste: 2.2 (antes 1.5)
  - Nitidez: 2.5 (antes 2.0)
  - Brilho: 1.15
  - Saturação: 1.3 (novo)

### 3. **OCR com Múltiplas Estratégias**
5 tentativas sequenciais com diferentes pré-processamentos:

1. **Direct** - Imagem como recebida
2. **Adaptive Threshold** - Binarização adaptativa (Gaussian)
3. **OTSU Threshold** - Binarização por OTSU
4. **Inverted** - Imagem invertida (para placas escuras)
5. **High-Contrast OTSU** - Contraste alto + OTSU

Cada tentativa usa `allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'`

### 4. **Correção Automática de Erros OCR**
Converte confusões comuns de OCR:
- `O` (letra O) → `0` (zero)
- `I` → `1` (um)
- `L` → `1` (um)
- `S` → `5` (cinco)
- `Z` → `2` (dois)
- `B` → `8` (oito)
- `G` → `9` (nove)

Válido apenas nas posições esperadas (dígitos: posições 3-6)

### 5. **Validação Flexível**
- Aceita separadores: `-`, espaço
- Valida formatos:
  - Antigo: `AAA1234` ou `AAA-1234` ou `AAA 1234`
  - Mercosul: `AAA1A23`

### 6. **Logs Detalhados para Debug**
Com `DEBUG=True`, mostra:
```
✓ ROI ativada com pontos: [(580, 230), (920, 150), (980, 520), (540, 600)]
Carro 0: coordenadas (100, 50, 300, 400)
  ✓ Carro dentro da ROI
  ✓ Processando detecção de placas...
  ✓ 1 placa(s) detectada(s)
  [✓] OCR resultado: 'ABC1234' (confiança: 0.95)
  [✓] Validação: 'ABC1234' -> 'ABC1234'
  ✓✓ PLACA ACEITA: ABC 1234
```

## 🚀 Como Testar

### Opção 1: Usar o Vídeo de Teste
```bash
cd /Users/felipefranco/alpr-camera/main-gate-alpr/server
source ../.venv/bin/activate
DEBUG=True python server.py
```

Aguarde até o ônibus passar na frente da câmera (no vídeo).

### Opção 2: Usar Câmera RTSP Real
Edite `.env`:
```bash
RTSP_CAPTURE_CONFIG="rtsp://usuario:senha@IP:554/Streaming/channels/1/"
```

### Verificar Imagens Intermediárias
```bash
# Ver imagens de diagnóstico
ls -lh intermediate_detection_files/
open intermediate_detection_files/latest_frame_with_roi.jpg
```

## 📊 Configurações Ajustáveis

No `.env`:
```bash
# Confiança mínima de detecção (0.0-1.0)
# Aumentar = menos falsos positivos, menos detecções
# Diminuir = mais detecções, mais falsos positivos
YOLO_CONFIDENCE=0.45

# Mínimo de caracteres para aceitar OCR
MINIMUM_NUMBER_OF_CHARS_FOR_MATCH=4

# Quantas vezes detectar antes de validar
NUMBER_OF_VALIDATION_ROUNDS=3

# Quantas ocorrências iguais para aceitar
NUMBER_OF_OCCURRENCES_TO_BE_VALID=2

# Distância mínima (y_max) para processar
SKIP_BEFORE_Y_MAX=0

# Pontos da ROI (quadrilátero)
ROI_POINTS=580,230;920,150;980,520;540,600
```

## 🎯 Próximos Passos se ainda não funcionar

### 1. Aumentar o modelo
```bash
# Mudar de yolov8m para yolov8l (mais lento, mais preciso)
PURE_YOLO_MODEL_PATH="../ai/resources/yolov8l.pt"
```

### 2. Aumentar tolerância OCR
```bash
# Aceitar placas com só 3 caracteres (mais liberal)
MINIMUM_NUMBER_OF_CHARS_FOR_MATCH=3
```

### 3. Recalibrar ROI
Se o ônibus não entra totalmente na ROI:
```bash
python calibrate_roi.py
```

### 4. Verificar condições de luz
Adicionar mais brilho ao enhance:
```python
# Em utils.py, na função enhance_plate_image()
enhancer = ImageEnhance.Brightness(pil_img)
pil_img = enhancer.enhance(1.30)  # Aumentar de 1.15 para 1.30
```

## 📝 Resumo das Mudanças
- ✅ Modelo YOLO de detecção: `yolov8n` → `yolov8m` 
- ✅ Pré-processamento: CLAHE aprimorado + Enhance agressivo
- ✅ OCR: 5 estratégias diferentes
- ✅ Validação: Flexível com correção de erros
- ✅ Logs: Muito mais detalhados para debug
- ✅ ROI: Polígono customizável já calibrada

