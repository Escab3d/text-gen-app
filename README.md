# Text Generation Studio (FastAPI + Streamlit + PyTorch)

An end-to-end web application and REST API serving a fine-tuned text generation language model (~300M parameters) using PyTorch, Hugging Face Transformers, FastAPI, and Streamlit.

---

## 🏛️ System Architecture

```text
+-----------------------+             HTTP POST /generate             +-----------------------+
|  Streamlit Frontend   | ------------------------------------------> |    FastAPI Backend    |
|  (Port 8501)          | <------------------------------------------ |    (Port 8000)        |
+-----------------------+              JSON Response                  +-----------------------+
                                                                                  |
                                                                           Model Inference
                                                                                  v
                                                                      +-----------------------+
                                                                      | HuggingFace / PyTorch |
                                                                      | AutoModelForCausalLM  |
                                                                      +-----------------------+
```

* **Backend (FastAPI):** High-performance asynchronous REST API handling input validation with Pydantic, device management (CUDA/CPU), tokenization, and generation.
* **Frontend (Streamlit):** Reactive web interface with real-time health checks, interactive sliders for generation parameters, and live inference speed metrics (tokens/second, latency).

---

## 📁 Repository Structure

```text
text-gen-app/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application & route endpoints
│   │   ├── schemas.py       # Pydantic request/response models
│   │   └── model_loader.py  # ModelService class managing weights & generation
│   ├── requirements.txt
│   └── model_weights/       # (Ignored by git) Place model.safetensors & config here
├── frontend/
│   ├── app.py               # Streamlit interactive UI
│   └── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Place Your Model Weights
Place your model folder (containing `model.safetensors`, `config.json`, `tokenizer.json`, etc.) into:
`backend/model_weights/`

Alternatively, you can set an environment variable pointing to any custom directory:
```bash
# Windows PowerShell
$env:MODEL_PATH="C:\path\to\your\model_folder"
```

---

### 2. Set Up Virtual Environment & Dependencies

Open a terminal in the root directory:

```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r backend/requirements.txt

# Install frontend dependencies
pip install -r frontend/requirements.txt
```

---

### 3. Start the Services

**Terminal 1: Start FastAPI Backend**
```bash
cd backend
python -m app.main
```
> The API will be live at `http://localhost:8000`. You can inspect and test the interactive API docs at `http://localhost:8000/docs`.

**Terminal 2: Start Streamlit Frontend**
```bash
cd frontend
streamlit run app.py
```
> The web interface will open in your browser at `http://localhost:8501`.

---

## 📡 API Reference

### Health Check
* **Endpoint:** `GET /health`
* **Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### Text Generation
* **Endpoint:** `POST /generate`
* **Request Body:**
```json
{
  "prompt": "The future of artificial intelligence is",
  "max_new_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9
}
```
* **Response:**
```json
{
  "prompt": "The future of artificial intelligence is",
  "generated_text": " going to revolutionize how we build software...",
  "tokens_generated": 50,
  "latency_seconds": 0.42
}
```

---

## 📊 Key Engineering Decisions
* **Model Lifetime Management:** Models are loaded into memory once on server startup using FastAPI's `lifespan` context manager, preventing multi-second reloading overhead on every request.
* **Precision Handling:** Uses `torch.float16` when CUDA is available to halve memory consumption and accelerate matrix multiplication, falling back gracefully to `torch.float32` on CPU.
* **Separation of Concerns:** Model loading and PyTorch tensors are strictly decoupled from API routing logic, making it straightforward to swap models or add batching in the future.
