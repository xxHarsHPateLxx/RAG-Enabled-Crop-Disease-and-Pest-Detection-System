# AgriSentry - AI Crop Disease Detection System

An end-to-end AI diagnostic tool for crop disease detection using CNN models and RAG (Retrieval-Augmented Generation) for intelligent treatment recommendations.

## 🌾 Features

- **Multimodal Disease Detection**: Optional vision-language model support with CNN fallback for Wheat, Rice, and Maize
- **Agentic RAG System**: Retrieval-Augmented Generation that can refine its search before answering
- **Conversation Memory**: Session-aware follow-up questions and diagnosis history
- **Structured Reasoning**: Internal severity, confidence, and intervention assessment before advice generation
- **Real-time Analysis**: Fast image processing and prediction
- **User-Friendly Interface**: Modern React-based web interface
- **Mobile Support**: Camera capture support for mobile devices

## 🏗️ Architecture

### Frontend (`/client`)
- React + Vite
- Tailwind CSS for styling
- React Router for navigation
- Lucide React for icons

### Backend (`/server`)
- FastAPI server
- Optional multimodal vision model adapter with CNN fallback
- TensorFlow CNN models (224x224 input)
- FAISS vector store for knowledge base
- Ollama Cloud models for structured reasoning, retrieval planning, treatment generation, and multimodal image analysis
- Session memory store for follow-up conversations
- LangChain for RAG orchestration

## 📋 Prerequisites

- Python 3.10+
- Node.js 18+
- Ollama Cloud API key

## 🚀 Setup & Installation

### Backend Setup

1. Navigate to server directory:
```bash
cd server
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```bash
cp .env.example .env
```

4. Add your Ollama API key to `.env`:
```
OLLAMA_API_KEY=your_api_key_here
```

5. Configure Ollama Cloud for both text and multimodal analysis:
```
TEXT_LLM_PROVIDER=ollama
TEXT_LLM_MODEL=deepseek-v3.2:cloud
TEXT_LLM_BASE_URL=https://ollama.com/api
VISION_LLM_PROVIDER=ollama
VISION_LLM_BASE_URL=https://ollama.com/api
VISION_LLM_MODEL=gemini-3-flash-preview:cloud
```

6. Initialize FAISS vector store (first time only):
```bash
python vectorize_kbase.py
```

7. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to client directory:
```bash
cd client
```

2. Install dependencies:
```bash
npm install
```

3. Run development server:
```bash
npm run dev
```

The app will be available at `http://localhost:5173`

## 🐳 Docker Deployment

Build and run using Docker:

```bash
# Build image
docker build -t agrisentry .

# Run container
docker run -p 7860:7860 -e MISTRAL_API_KEY=your_api_key agrisentry
```

## 📁 Project Structure

```
mpr/
├── client/                # React frontend
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   ├── pages/         # Route pages
│   │   └── App.jsx        # Main app component
│   └── package.json
├── server/                # FastAPI backend
│   ├── models/            # CNN model files (.h5)
│   ├── faiss_index/       # FAISS vector store
│   ├── main.py            # API server
│   ├── vectorize_kbase.py # Vector store builder
│   ├── kbase.json         # Knowledge base
│   └── requirements.txt
├── Dockerfile             # Production deployment
└── README.md
```

## 🌱 Supported Crops & Diseases

### Wheat
- Smut
- Leaf Blight
- Brown Rust
- Healthy

### Rice
- Bacterial Leaf Blight
- Brown Spot
- Leaf Blast
- Healthy

### Maize
- Blight
- Common Rust
- Gray Leaf Spot
- Healthy

## 🔧 API Endpoints

### `POST /api/predict`
Upload crop image and get disease prediction with treatment advice.

**Request:**
- `file`: Image file (multipart/form-data)
- `crop`: Crop type (Wheat/Rice/Maize)

**Response:**
```json
{
  "session_id": "d7b5c1d4-0b8d-4a6d-9a2e-4c5fd5c3b1f1",
  "crop": "Wheat",
  "disease": "Brown Rust",
  "confidence": 0.95,
  "analysis_source": "multimodal_llm",
  "reasoning": {
    "severity_assessment": "high",
    "confidence_band": "high",
    "intervention_mode": "integrated",
    "intervention_priority": "chemical and cultural"
  },
  "advice": "Treatment recommendations...",
  "follow_up_questions": ["How urgent is this?", "What if I cannot find the preferred product?"]
}
```

### `POST /api/chat`
Continue a diagnosis conversation using the stored session memory.

**Request:**
```json
{
  "session_id": "d7b5c1d4-0b8d-4a6d-9a2e-4c5fd5c3b1f1",
  "message": "What if I can only get a different fungicide?"
}
```

**Response:**
```json
{
  "session_id": "d7b5c1d4-0b8d-4a6d-9a2e-4c5fd5c3b1f1",
  "answer": "..."
}
```

## 🧪 Model Details

- **Architecture**: Multimodal vision analysis when configured, with CNN fallback
- **Input Size**: 224x224 RGB images
- **Output**: 4 classes per crop (3 diseases + Healthy)
- **Framework**: TensorFlow/Keras

## 📊 Knowledge Base

The RAG system uses:
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **LLM**: Ollama Cloud (`deepseek-v3.2:cloud` + `gemini-3-flash-preview:cloud`)
- **Memory**: JSON-backed session store for follow-up context
- **Source**: Structured disease information in `kbase.json`

## 🛠️ Development

### Adding New Diseases

1. Update `kbase.json` with new disease information
2. Retrain the CNN model or configure the multimodal provider for the new crop
3. Run `python vectorize_kbase.py` to update FAISS index
4. Update `MODEL_PATHS` and `LABELS` in `main.py`

### Building for Production

Frontend:
```bash
cd client
npm run build
```

Backend (Docker):
```bash
docker build -t agrisentry .
docker run -p 7860:7860 agrisentry
```



## ⚠️ Notes

- Ensure `.env` file is created with a valid `OLLAMA_API_KEY`
- Keep `TEXT_LLM_PROVIDER=ollama` and `VISION_LLM_PROVIDER=ollama`
- Model files (~159MB) should be placed in `server/models/`
- FAISS index must be generated before first run
- API key should never be committed to version control
