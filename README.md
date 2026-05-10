## Run Backend

### Requirements

```bash
pip install fastapi uvicorn torch mediapipe opencv-python sentencepiece
```

### Start the API

```bash
python api.py
```

The API will be available at `http://localhost:5000`

## Run Frontend

```bash
cd ifhamnii
npm install
npm run dev
```

The app will be available at `http://localhost:5173`

## Requirements

### Backend

- Python 3.10+
- CUDA-compatible GPU (recommended)
- See `requirements.txt` for full dependencies

### Frontend

- Node.js 18+
