# ALPR Gate - Sistema de Reconhecimento de Placas

Sistema de monitoramento e leitura automática de placas de veículos (ALPR) otimizado para o padrão brasileiro (Antigo e Mercosul), com integração PostgreSQL e interface em Rust.

## 🚀 Funcionalidades

- **Reconhecimento Inteligente:** Detecta placas brasileiras via YOLOv8.
- **Padrão Brasileiro:** Filtro estrito para formatos `AAA-1234` e `AAA1A23`.
- **Banco de Dados:** Armazenamento em PostgreSQL via Docker.
- **Interface em Rust:** Cliente leve com visualização em tempo real via WebSockets.
- **Limpeza Automática:** Remoção de logs e imagens locais após 1 dia.
- **Otimizado para Mac:** Suporte nativo para chips Apple Silicon (M1/M2/M3).

##  Como funciona?

1. Quando você inicia o servidor web, todas as variáveis de ambiente e modelos de IA são carregados na memória.
2. Se o salvamento de resultados estiver ativado, o diretório de resultados será criado.
3. Duas threads são iniciadas:
   - A primeira lê os quadros (frames) da câmera IP ou vídeo e garante a conexão constante com a fonte. Em modo `DEBUG`, uma janela de visualização será aberta.
4. A segunda thread obtém o quadro mais recente e o passa para o modelo YOLO principal.
5. Após a análise, o programa recorta as imagens dos carros e verifica se o veículo não está muito longe (conforme `SKIP_BEFORE_Y_MAX`).
6. Em seguida, passa a imagem recortada para o modelo YOLO ajustado especificamente para placas.
7. A placa é recortada e pré-processada (veja detalhes em `./utils.py`).
8. A placa é então separada em cada caractere, que é passado ao Tesseract usando todas as threads disponíveis.
9. O valor da placa é finalizado e validado.
10. A placa e a imagem do carro são enviadas para todos os clientes conectados via WebSocket.
    - Opcionalmente, os dados são salvos no Banco de Dados ou na pasta de resultados, conforme seu `.env`.

## 🛠️ Como Configurar

### Pré-requisitos
- Python 3.11.2 (ou superior)
- Docker (opcional, para banco de dados local)
- Rust (opcional, para rodar o cliente GUI fornecido)

### 1. Preparação dos Modelos (Extração)
Vá para a pasta `./ai/resources` e execute os comandos para unir as partes dos modelos:
```bash
cat yolov8m_* > yolov8m.pt
cat yolov8l_* > yolov8l.pt
cat yolov8x_* > yolov8x.pt
cat andrewmvd_dataset_* > andrewmvd_dataset.zip
cat aslanahmedov_dataset_* > aslanahmedov_dataset.zip
cat tdiblik_lp_finetuned_yolov8m_* > tdiblik_lp_finetuned_yolov8m.pt
cat tdiblik_lp_finetuned_yolov8l_* > tdiblik_lp_finetuned_yolov8l.pt
cat tdiblik_lp_finetuned_yolov8x_* > tdiblik_lp_finetuned_yolov8x.pt
cp yolov8*.pt ..
```

### 2. Instalação de Dependências
Na raiz do projeto:
1. Instale o PyTorch (recomenda-se seguir as instruções do [site oficial para sua placa de vídeo](https://pytorch.org/)):
   `pip install torch torchvision torchaudio`
2. Instale o [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html).
3. Instale os requisitos do Python:
   `pip install -r requirements.txt`

### 3. Iniciar o Servidor (WebSocket e Processamento)
1. Vá para `./server`.
2. Copie `.env.development` para `.env` e configure suas variáveis.
3. Configure o banco de dados PostgreSQL (via Docker):
   ```bash
   docker run --name main_gate_aplr_db \
    -e POSTGRES_PASSWORD='MinhaSenhaForte' \
    -e POSTGRES_DB=lpdb \
    -p 5432:5432 \
    -v $(pwd)/db/data:/var/lib/postgresql/data \
    -d postgres:15-alpine
   ```
4. Inicialize o esquema do banco:
   `cat db/init.sql | docker exec -i main_gate_aplr_db psql -U postgres -d lpdb`
5. Execute o servidor:
   `python server.py`

### 4. Iniciar o Cliente de Exemplo (Opcional)
1. Vá para `./client`.
2. Verifique o `WEBSOCKET_URL` em `./client/src/main.rs`.
3. Execute: `cargo run`

## 🧠 Treinamento e Testes

### Treinar seu próprio modelo (Opcional)
1. Vá para a pasta `./ai`.
2. Execute `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` (no Linux/Mac) ou `set...` no Windows.
3. `python prepare.py`.
4. Configure em `train.py` qual modelo pré-treinado usar.
5. Inicie com `python train.py`.

### Testar modelos visualmente (Opcional)
1. Vá para a pasta `./ai`.
2. Execute: `python test.py {caminho_do_modelo} {caminho_da_imagem}`.

## ⚙️ Configurações do .env

### Base
- **DEBUG:** `True` para ver logs detalhados e abrir a janela de visualização do vídeo.
- **WS_PORT:** Porta para o servidor WebSocket.
- **RTSP_CAPTURE_CONFIG:** Link da câmera RTSP ou caminho do arquivo de vídeo (ex: `./test.mp4`).
- **PURE_YOLO_MODEL_PATH:** Modelo YOLO padrão para detectar carros.
- **LICENSE_PLATE_YOLO_MODEL_PATH:** Modelo ajustado para detectar placas.
- **DB_ENABLED:** `True` para salvar no banco de dados.
- **SAVE_RESULTS_ENABLED:** `True` para salvar imagens dos carros e placas detectadas.
- **SHOULD_SEND_SAME_RESULTS:** Define se deve ignorar a mesma placa se detectada repetidamente em 5 minutos.

### Ajustes Finos
- **SHOULD_TRY_LP_CROP:** Tenta recortar bordas extras da placa programaticamente.
- **MINIMUM_NUMBER_OF_CHARS_FOR_MATCH:** Mínimo de caracteres detectados (padrão 4).
- **NUMBER_OF_VALIDATION_ROUNDS:** Quantidade de frames para validar uma placa (padrão 3).
- **SKIP_BEFORE_Y_MAX:** Ignora carros que estão muito longe no topo da imagem para economizar CPU.

## 📝 Notas de Desenvolvimento
- O projeto foi migrado de MSSQL para PostgreSQL para melhor performance e compatibilidade.
- A detecção ignora automaticamente qualquer texto que não siga o padrão de placas do Brasil.
- Certifique-se de configurar o caminho do vídeo/câmera no seu arquivo `.env`.
- Ao adicionar recursos maiores que 25MB, use o comando `split -b 25M --numeric-suffixes <nome> <nome>_` para manter o Git leve.
- `lp` no código é uma abreviação de "License Plate" (Placa de Veículo).
- Atualmente, a maior limitação é o motor de OCR (Tesseract), que pode falhar ocasionalmente (cerca de 1 em cada 20 carros).

## 🙏 Agradecimentos
- Modelos YOLOv8 da [Ultralytics](https://github.com/ultralytics/ultralytics).
- Datasets de placas de [Andrew MVD](https://makeml.app/datasets/cars-license-plates) e [Aslan Ahmedov](https://www.kaggle.com/aslanahmedov).
- Tutorial YOLOv8 do [FreeCodeCamp](https://www.freecodecamp.org/news/how-to-detect-objects-in-images-using-yolov8/).

---
*Desenvolvido para automação de portarias e segurança.*
