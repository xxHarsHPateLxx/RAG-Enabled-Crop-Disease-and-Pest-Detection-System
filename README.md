# AgriSentry - AI Crop Disease Detection System

An end-to-end AI diagnostic tool for crop disease detection using CNN models and RAG (Retrieval-Augmented Generation) for intelligent treatment recommendations.

## 🌾 Features

- **CNN-based Disease Detection**: Deep learning models for Wheat, Rice, and Maize
- **RAG System**: Retrieval-Augmented Generation for personalized treatment advice
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
- TensorFlow CNN models (224x224 input)
- FAISS vector store for knowledge base
- Mistral AI for treatment generation
- LangChain for RAG orchestration

## 📋 Prerequisites

- Python 3.10+
- Node.js 18+
- Mistral AI API key

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

4. Add your Mistral API key to `.env`:
```
MISTRAL_API_KEY=your_api_key_here
```

5. Initialize FAISS vector store (first time only):
```bash
python vectorize_kbase.py
```

6. Run the server:
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
  "crop": "Wheat",
  "disease": "Brown Rust",
  "confidence": 0.95,
  "advice": "Treatment recommendations..."
}
```

## 🧪 Model Details

- **Architecture**: CNN (Convolutional Neural Network)
- **Input Size**: 224x224 RGB images
- **Output**: 4 classes per crop (3 diseases + Healthy)
- **Framework**: TensorFlow/Keras

## 📊 Knowledge Base

The RAG system uses:
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **LLM**: Mistral Small Latest
- **Source**: Structured disease information in `kbase.json`

## 🛠️ Development

### Adding New Diseases

1. Update `kbase.json` with new disease information
2. Retrain CNN model or add new crop model
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

- Ensure `.env` file is created with valid Mistral API key
- Model files (~159MB) should be placed in `server/models/`
- FAISS index must be generated before first run
- API key should never be committed to version control
