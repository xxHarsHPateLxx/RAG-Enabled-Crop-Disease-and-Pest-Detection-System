import io
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
import tensorflow as tf
import importlib
import os
import re
from dotenv import load_dotenv



# Load variables from .env
load_dotenv()

# Access variables
mistral_key = os.getenv("MISTRAL_API_KEY")

# ------------------
# CONFIG
# -------------------
FAISS_STORE_PATH = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MISTRAL_MODEL = "mistral-small-latest"  # Or mistral-small-latest

MODEL_PATHS = {
    "Wheat": "models/wheat_cnn.h5",
    "Rice": "models/rice_cnn.h5",
    "Maize": "models/maize_cnn.h5"
}

LABELS = {
    "Wheat": {0: "Smut", 1: "Leaf Blight", 2: "Brown Rust", 3: "Healthy"},
    "Rice": {0: "Bacterial Leaf Blight", 1: "Brown Spot", 2: "Leaf Blast", 3: "Healthy"},
    "Maize": {0: "Blight", 1: "Common Rust", 2: "Gray Leaf Spot", 3: "Healthy"}
}

IMG_SIZE = (224, 224)

# -------------------
# INIT APP
# -------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Load embeddings & vectorstore
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vector_store = FAISS.load_local(
    FAISS_STORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# Load Mistral API LLM
llm = init_chat_model(MISTRAL_MODEL, model_provider="mistralai")

# -------------------
# HELPERS
# -------------------
def load_model(crop_name: str):
    """Force load using TF's built-in keras."""
    tf_keras_models = importlib.import_module("tensorflow.keras.models")
    return tf_keras_models.load_model(MODEL_PATHS[crop_name], compile=False)

def predict(model, idx_to_class, image: Image.Image):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    return idx_to_class[class_idx], confidence


def clean_json_output(text: str):
    # Remove markdown code block formatting if present
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)  # Remove starting ```json or ```
    text = re.sub(r"\n```$", "", text)           # Remove ending ```
    text = text.strip()
    return text


# -------------------
# API
# -------------------
@app.post("/api/predict")
async def predict_and_rag(
    crop: str = Form(...),
    file: UploadFile = File(...)
):
    # Load relevant CNN
    model = load_model(crop)

    # Read image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Predict disease
    disease, confidence = predict(model, LABELS[crop], image)

    # Retrieve relevant docs from FAISS
    query = f"Crop: {crop}, Disease: {disease}"
    docs = vector_store.similarity_search(query, k=3)
    context_text = "\n\n".join([d.page_content for d in docs])

    # Create prompt for LLM
    template = """
You are an agricultural expert. Based on the following crop disease prediction:

Crop: {crop}
Disease: {disease}
Confidence: {confidence}

Here is relevant info from the knowledge base:
{context}

Provide a comprehensive, farmer-friendly analysis with the following sections. Format your response using markdown:

**Description**
Write a detailed explanation of the disease in 3-4 clear sentences. Explain what it is, how it appears, and its impact on crops.

**Cause**
Explain what causes this disease. Use bullet points (start lines with -) to list:
- The primary pathogen or cause
- Environmental conditions that favor the disease
- How the disease spreads

**Treatment**
Provide specific, actionable treatment recommendations. Use bullet points (start lines with -) for each treatment step:
- Chemical treatments (specific fungicides/pesticides with application rates)
- Cultural practices (pruning, spacing, etc.)
- Biological controls if applicable
- Timing and frequency of treatments

**Prevention**
List preventive measures farmers can take. Use bullet points (start lines with -):
- Pre-planting practices
- Crop management techniques
- Resistant varieties if available
- Monitoring and early detection methods

Write in a clear, professional tone that's easy for farmers to understand. Use proper spacing between sections.
"""
    prompt = PromptTemplate(
        input_variables=["crop", "disease", "confidence", "context"],
        template=template
    )

    final_prompt = prompt.format(
        crop=crop.capitalize(),
        disease=disease,
        confidence=round(confidence, 4),
        context=context_text
    )

    # Get response from Mistral API
    response = llm.invoke(final_prompt)
    raw_text = response.content if hasattr(response, "content") else str(response)

    return {
        "crop": crop,
        "disease": disease,
        "confidence": round(confidence, 4),
        "advice": raw_text
    }

# -------------------
# RUN
# -------------------
if __name__ == "__main__":
    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError("❌ MISTRAL_API_KEY not set. Please export it before running.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
