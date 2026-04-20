from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from PIL import Image
from pydantic import BaseModel, Field

from knowledge_base_utils import build_documents_from_knowledge_base
from memory_store import MemoryStore


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
FAISS_STORE_PATH = BASE_DIR / "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KBASE_FILE = BASE_DIR / "kbase.json"
SESSION_STORE_PATH = BASE_DIR / "session_memory.json"
TEXT_LLM_PROVIDER = os.getenv("TEXT_LLM_PROVIDER", "ollama").strip().lower()
TEXT_LLM_MODEL = os.getenv("TEXT_LLM_MODEL", "deepseek-v3.2:cloud")
TEXT_LLM_BASE_URL = os.getenv(
    "TEXT_LLM_BASE_URL",
    "https://ollama.com/api",
).rstrip("/")
TEXT_LLM_API_KEY = (
    os.getenv("TEXT_LLM_API_KEY")
    or os.getenv("OLLAMA_API_KEY")
)
VISION_PROVIDER = os.getenv("VISION_LLM_PROVIDER", "ollama").strip().lower()
VISION_MODEL = os.getenv("VISION_LLM_MODEL", "gemini-3-flash-preview:cloud")
VISION_API_KEY = (
    os.getenv("VISION_LLM_API_KEY")
    or os.getenv("OLLAMA_API_KEY")
)
VISION_BASE_URL = os.getenv(
    "VISION_LLM_BASE_URL",
    "https://ollama.com/api",
).rstrip("/")

MODEL_PATHS = {
    "Wheat": BASE_DIR / "models" / "wheat.h5",
    "Rice": BASE_DIR / "models" / "rice.h5",
    "Maize": BASE_DIR / "models" / "maize.h5",
}

LABELS = {
    "Wheat": {0: "Smut", 1: "Leaf Blight", 2: "Brown Rust", 3: "Healthy"},
    "Rice": {0: "Bacterial Leaf Blight", 1: "Brown Spot", 2: "Leaf Blast", 3: "Healthy"},
    "Maize": {0: "Blight", 1: "Common Rust", 2: "Gray Leaf Spot", 3: "Healthy"},
}

IMG_SIZE = (224, 224)
MAX_RETRIEVAL_ROUNDS = 2
OLLAMA_PROVIDERS = {"ollama", "ollama-cloud", "ollama_cloud"}


class ReasoningPayload(BaseModel):
    severity_assessment: str = Field(default="moderate")
    confidence_band: str = Field(default="medium")
    intervention_mode: str = Field(default="integrated")
    intervention_priority: str = Field(default="cultural and chemical")
    key_factors: list[str] = Field(default_factory=list)
    evidence_points: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    summary: str = Field(default="")


class ChatRequest(BaseModel):
    session_id: str
    message: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory_store = MemoryStore(SESSION_STORE_PATH)


def build_faiss_index():
    documents = build_documents_from_knowledge_base(KBASE_FILE)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(str(FAISS_STORE_PATH))
    return embeddings, vector_store


embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
if FAISS_STORE_PATH.exists():
    vector_store = FAISS.load_local(
        str(FAISS_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    embeddings, vector_store = build_faiss_index()


def load_model(crop_name: str):
    model_path = MODEL_PATHS[crop_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    try:
        from keras.models import load_model as keras_load_model
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow/Keras is not available in this environment, so CNN predictions cannot run."
        ) from exc

    return keras_load_model(str(model_path), compile=False)


def predict_with_cnn(model, idx_to_class, image: Image.Image) -> tuple[str, float]:
    resized = image.resize(IMG_SIZE)
    image_array = np.array(resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array, verbose=0)
    class_idx = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))
    return idx_to_class[class_idx], confidence


def clean_json_output(text: str) -> str:
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", text)
    cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def safe_json_loads(text: str) -> dict[str, Any] | None:
    cleaned = clean_json_output(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    if confidence > 1.0:
        confidence = confidence / 100.0
    return max(0.0, min(confidence, 1.0))


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_http_error_message(error: urllib.error.HTTPError) -> str:
    try:
        payload = error.read().decode("utf-8")
        parsed = json.loads(payload)
        message = parsed.get("error", {}).get("message")
        if message:
            return str(message)
        return payload.strip() or f"HTTP {error.code}"
    except Exception:
        return f"HTTP {error.code}"


def build_ollama_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def call_ollama_text_model(prompt: str, expect_json: bool = False) -> str | None:
    if TEXT_LLM_PROVIDER not in OLLAMA_PROVIDERS:
        return None

    payload: dict[str, Any] = {
        "model": TEXT_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    if expect_json:
        payload["format"] = "json"

    request = urllib.request.Request(
        f"{TEXT_LLM_BASE_URL}/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers=build_ollama_headers(TEXT_LLM_API_KEY),
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw_response = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Ollama text call failed: {extract_http_error_message(error)}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError("Ollama text call failed: network or response parsing error.") from error

    content = ((raw_response.get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError("Ollama text call failed: empty response from model.")

    return content


def invoke_text_llm(prompt: str, expect_json: bool = False) -> str:
    if TEXT_LLM_PROVIDER in OLLAMA_PROVIDERS:
        if TEXT_LLM_BASE_URL.startswith("https://ollama.com") and not TEXT_LLM_API_KEY:
            raise RuntimeError("Ollama cloud API key is missing. Set OLLAMA_API_KEY.")
        return call_ollama_text_model(prompt, expect_json=expect_json)

    raise RuntimeError("Unsupported text provider. Use TEXT_LLM_PROVIDER=ollama.")


class OllamaTextBridgeLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "ollama-http-bridge"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> str:
        response = invoke_text_llm(prompt, expect_json=False)
        if stop:
            for stop_token in stop:
                if stop_token in response:
                    response = response.split(stop_token, 1)[0]
        return response


REACT_AGENT_PROMPT = PromptTemplate.from_template(
    """
You are an agricultural retrieval agent. Your job is to fetch relevant crop-disease context.

Rules:
- Always call a retrieval tool first.
- You may refine and retry with a better query if context is incomplete.
- Keep retrieval focused on crop, disease, treatment, prevention, severity, and safety.
- Use at most the available tool iterations.
- In your final answer, return only the best retrieved context for downstream generation.

You have access to the following tools:
{tools}

Use this format:
Question: the input question
Thought: your reasoning
Action: one of [{tool_names}]
Action Input: query text for retrieval
Observation: tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have the context
Final Answer: consolidated retrieved context

Question: {input}
Thought:{agent_scratchpad}
""".strip()
)

text_agent_llm = OllamaTextBridgeLLM()


def call_ollama_vision_model(image: Image.Image, crop_name: str | None = None) -> dict[str, Any] | None:
    if not VISION_PROVIDER:
        return None

    if VISION_PROVIDER not in OLLAMA_PROVIDERS:
        return None

    if crop_name:
        prompt = (
            "You are diagnosing a crop disease from an image. "
            f"Crop: {crop_name}. "
            "Return JSON only with keys: crop, disease, confidence, severity_hint, visual_summary, evidence_points. "
            "Keep crop exactly as provided. Use the disease label that best matches the image. "
            "Confidence must be a number between 0 and 1."
        )
    else:
        prompt = (
            "You are diagnosing a crop disease from an image. "
            "Infer the crop type and disease from the image. "
            "Supported crops are Wheat, Rice, and Maize, but if uncertain return the closest crop guess. "
            "Return JSON only with keys: crop, disease, confidence, severity_hint, visual_summary, evidence_points. "
            "Confidence must be a number between 0 and 1."
        )

    image_b64 = image_to_data_url(image).split(",", 1)[1]
    payload: dict[str, Any] = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }

    request = urllib.request.Request(
        f"{VISION_BASE_URL}/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers=build_ollama_headers(VISION_API_KEY),
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw_response = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Ollama vision call failed: {extract_http_error_message(error)}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError("Ollama vision call failed: network or response parsing error.") from error

    content = ((raw_response.get("message") or {}).get("content") or "").strip()
    parsed = safe_json_loads(content) or {}
    disease = parsed.get("disease")
    if not disease:
        raise RuntimeError("Ollama vision call failed: model returned no disease label.")

    inferred_crop = str(parsed.get("crop") or crop_name or "Unknown").strip().title()
    return {
        "analysis_source": "multimodal_llm",
        "crop": inferred_crop,
        "disease": str(disease),
        "confidence": normalize_confidence(parsed.get("confidence", 0.7)),
        "visual_summary": str(parsed.get("visual_summary", "")),
        "severity_hint": str(parsed.get("severity_hint", "moderate")),
        "evidence_points": parsed.get("evidence_points", []),
    }


def dedupe_documents(documents):
    seen: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
    unique_documents = []

    for document in documents:
        metadata_items = tuple(sorted((document.metadata or {}).items()))
        fingerprint = (document.page_content, metadata_items)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique_documents.append(document)

    return unique_documents


def format_documents(documents) -> str:
    formatted_sections: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        section = [f"Source {index}"]
        if metadata.get("crop"):
            section.append(f"Crop: {metadata['crop']}")
        if metadata.get("disease"):
            section.append(f"Disease: {metadata['disease']}")
        section.append(document.page_content)
        formatted_sections.append("\n".join(section))
    return "\n\n---\n\n".join(formatted_sections)


def extract_assistance_score(confidence: float, disease: str) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"


def build_retrieval_query(crop: str, disease: str, memory_context: str, image_summary: str | None = None) -> str:
    parts = [f"Crop {crop}", f"Disease {disease}", "treatment", "prevention"]
    if image_summary:
        parts.append(image_summary)
    if memory_context:
        parts.append(memory_context)
    return ", ".join(part for part in parts if part)


def agentic_retrieve(crop: str, disease: str, confidence: float, memory_context: str, image_summary: str | None = None):
    collected_documents = []
    trace: list[dict[str, Any]] = []

    def retrieve_crop_knowledge(query: str) -> str:
        documents = vector_store.similarity_search(query, k=4)
        collected_documents.extend(documents)
        unique_documents = dedupe_documents(collected_documents)
        trace.append(
            {
                "round": len(trace) + 1,
                "query": query,
                "documents": [document.metadata for document in documents],
            }
        )
        return format_documents(unique_documents)

    tools = [
        Tool(
            name="retrieve_crop_knowledge",
            func=retrieve_crop_knowledge,
            description=(
                "Retrieve agricultural context using FAISS. "
                "Input should be a focused query with crop, disease, and agronomy intent."
            ),
        )
    ]

    seed_query = build_retrieval_query(crop, disease, memory_context, image_summary)
    agent_input = f"""
Crop: {crop}
Disease: {disease}
Model confidence: {confidence:.4f}
Image summary: {image_summary or 'Not available'}
Recent memory context:
{memory_context or 'None'}

Initial retrieval query:
{seed_query}
""".strip()

    try:
        from langchain_classic.agents import AgentExecutor, create_react_agent

        agent = create_react_agent(text_agent_llm, tools, REACT_AGENT_PROMPT)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=MAX_RETRIEVAL_ROUNDS,
        )
        agent_executor.invoke({"input": agent_input})
    except Exception:
        if not trace:
            retrieve_crop_knowledge(seed_query)
            if image_summary or memory_context:
                follow_up_query = build_retrieval_query(crop, disease, memory_context, image_summary)
                if follow_up_query != seed_query:
                    retrieve_crop_knowledge(follow_up_query)

    unique_documents = dedupe_documents(collected_documents)
    retrieval_context = format_documents(unique_documents)
    return unique_documents, trace, retrieval_context


def build_reasoning_payload(
    crop: str,
    disease: str,
    confidence: float,
    retrieval_context: str,
    memory_context: str,
    image_summary: str | None = None,
    severity_hint: str | None = None,
) -> ReasoningPayload:
    prompt = f"""
You are structuring an agricultural diagnosis for internal reasoning.

Crop: {crop}
Disease: {disease}
Image model confidence: {confidence:.4f}
Severity hint from vision model: {severity_hint or 'unknown'}
Recent memory context:
{memory_context or 'None'}

Retrieved knowledge:
{retrieval_context}

Image summary:
{image_summary or 'Not available'}

Return JSON only with keys:
- severity_assessment: string
- confidence_band: string
- intervention_mode: string
- intervention_priority: string
- key_factors: array of strings
- evidence_points: array of strings
- safety_notes: array of strings
- follow_up_questions: array of strings
- summary: string
"""

    raw_text = invoke_text_llm(prompt, expect_json=True)
    parsed = safe_json_loads(raw_text)
    if parsed:
        try:
            return ReasoningPayload(**parsed)
        except Exception:
            pass

    confidence_band = extract_assistance_score(confidence, disease)
    return ReasoningPayload(
        severity_assessment=severity_hint or ("high" if confidence >= 0.8 else "moderate"),
        confidence_band=confidence_band,
        intervention_mode="integrated" if confidence >= 0.55 else "cultural",
        intervention_priority="chemical and cultural" if confidence >= 0.55 else "monitoring and cultural",
        key_factors=[f"Model confidence: {confidence:.2f}", f"Diagnosis: {disease}"] + ([image_summary] if image_summary else []),
        evidence_points=["Retrieved knowledge base documents matched the predicted crop and disease."],
        safety_notes=["Validate treatment labels locally before application."],
        follow_up_questions=[
            "How urgent is this field condition?",
            "What treatment alternatives are available if the preferred product is not available?",
            "What should I monitor over the next 7 days?",
        ],
        summary=f"The crop shows signs consistent with {disease} and should be managed with a {confidence_band}-confidence response.",
    )


def generate_advice(
    crop: str,
    disease: str,
    confidence: float,
    reasoning: ReasoningPayload,
    retrieval_context: str,
    memory_context: str,
    image_summary: str | None = None,
) -> str:
    prompt = f"""
You are an agricultural advisor.

Crop: {crop}
Disease: {disease}
Confidence: {confidence:.4f}

Structured reasoning:
{reasoning.model_dump_json(indent=2)}

Recent memory context:
{memory_context or 'None'}

Retrieved knowledge:
{retrieval_context}

Image summary:
{image_summary or 'Not available'}

Write a clear farmer-facing answer in markdown with these sections:
## Assessment
## What It Means
## Treatment
## Prevention
## Follow-Up

Keep the advice concrete and brief. If you mention chemicals, include the need to confirm local label and extension guidance.
Use bullet points for actionable items.
"""

    return invoke_text_llm(prompt, expect_json=False)


def build_diagnosis_record(
    session_id: str,
    crop: str,
    disease: str,
    confidence: float,
    reasoning: ReasoningPayload,
    analysis_source: str,
    retrieval_trace: list[dict[str, Any]],
    image_summary: str | None,
) -> dict[str, Any]:
    diagnosis_id = str(uuid4())
    return {
        "session_id": session_id,
        "diagnosis_id": diagnosis_id,
        "crop": crop,
        "disease": disease,
        "confidence": round(confidence, 4),
        "analysis_source": analysis_source,
        "reasoning": reasoning.model_dump(),
        "retrieval_trace": retrieval_trace,
        "image_summary": image_summary or "",
    }


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict_and_rag(
    crop: str | None = Form(None),
    file: UploadFile = File(...),
    prediction_mode: str = Form("cnn_llm"),
    session_id: str | None = Form(None),
    memory_key: str | None = Form(None),
):
    mode = prediction_mode.strip().lower()
    if mode not in {"multimodal_llm", "cnn_llm"}:
        mode = "cnn_llm"

    normalized_crop = crop.strip().title() if crop else None

    resolved_session_id = memory_store.ensure_session(session_id)
    resolved_memory_key = memory_key or resolved_session_id

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    image_summary = None
    severity_hint = None
    analysis_source = "unavailable"

    if mode == "multimodal_llm":
        if VISION_PROVIDER in OLLAMA_PROVIDERS and VISION_BASE_URL.startswith("https://ollama.com") and not VISION_API_KEY:
            return {
                "session_id": resolved_session_id,
                "memory_key": resolved_memory_key,
                "prediction_mode": mode,
                "crop": normalized_crop or "Unknown",
                "disease": "Unavailable",
                "confidence": 0.0,
                "analysis_source": "unavailable",
                "advice": "Multimodal mode requires Ollama cloud API key. Set OLLAMA_API_KEY.",
            }

        if VISION_PROVIDER not in OLLAMA_PROVIDERS:
            return {
                "session_id": resolved_session_id,
                "memory_key": resolved_memory_key,
                "prediction_mode": mode,
                "crop": normalized_crop or "Unknown",
                "disease": "Unavailable",
                "confidence": 0.0,
                "analysis_source": "unavailable",
                "advice": "Unsupported vision provider. Use VISION_LLM_PROVIDER=ollama.",
            }

        try:
            vision_result = call_ollama_vision_model(image, normalized_crop)
            crop = str(vision_result.get("crop") or normalized_crop or "Unknown").strip().title()
            disease = vision_result["disease"]
            confidence = float(vision_result["confidence"])
            image_summary = vision_result.get("visual_summary")
            severity_hint = vision_result.get("severity_hint")
            analysis_source = vision_result.get("analysis_source", "multimodal_llm")
        except RuntimeError as exc:
            return {
                "session_id": resolved_session_id,
                "memory_key": resolved_memory_key,
                "prediction_mode": mode,
                "crop": normalized_crop or "Unknown",
                "disease": "Unavailable",
                "confidence": 0.0,
                "analysis_source": "unavailable",
                "advice": str(exc),
            }
    else:
        if not normalized_crop:
            return {
                "session_id": resolved_session_id,
                "memory_key": resolved_memory_key,
                "prediction_mode": mode,
                "crop": "Unknown",
                "disease": "Unavailable",
                "confidence": 0.0,
                "analysis_source": "unavailable",
                "advice": "CNN + LLM mode requires selecting a crop.",
            }

        if normalized_crop not in MODEL_PATHS:
            return {
                "session_id": resolved_session_id,
                "memory_key": resolved_memory_key,
                "prediction_mode": mode,
                "crop": normalized_crop,
                "disease": "Unavailable",
                "confidence": 0.0,
                "analysis_source": "unavailable",
                "advice": f"Unsupported crop '{normalized_crop}' for CNN mode.",
            }

        crop = normalized_crop
        try:
            model = load_model(crop)
            disease, confidence = predict_with_cnn(model, LABELS[crop], image)
            analysis_source = "cnn"
        except Exception as exc:
            can_use_ollama_fallback = (
                VISION_PROVIDER in OLLAMA_PROVIDERS
                and (not VISION_BASE_URL.startswith("https://ollama.com") or bool(VISION_API_KEY))
            )

            if can_use_ollama_fallback:
                try:
                    vision_result = call_ollama_vision_model(image, crop)
                    disease = vision_result["disease"]
                    confidence = float(vision_result["confidence"])
                    image_summary = vision_result.get("visual_summary")
                    severity_hint = vision_result.get("severity_hint")
                    analysis_source = "cnn_fallback_multimodal"
                except RuntimeError as vision_exc:
                    return {
                        "session_id": resolved_session_id,
                        "memory_key": resolved_memory_key,
                        "prediction_mode": mode,
                        "crop": crop,
                        "disease": "Unavailable",
                        "confidence": 0.0,
                        "analysis_source": "unavailable",
                        "advice": f"CNN path failed: {exc}. Multimodal fallback also failed: {vision_exc}",
                    }
            else:
                return {
                    "session_id": resolved_session_id,
                    "memory_key": resolved_memory_key,
                    "prediction_mode": mode,
                    "crop": crop,
                    "disease": "Unavailable",
                    "confidence": 0.0,
                    "analysis_source": "unavailable",
                    "advice": f"CNN path failed: {exc}. Multimodal fallback is not configured for the selected provider.",
                }

    try:
        memory_context = memory_store.recent_context(resolved_memory_key)
        retrieval_documents, retrieval_trace, retrieval_context = agentic_retrieve(
            crop=crop,
            disease=disease,
            confidence=confidence,
            memory_context=memory_context,
            image_summary=image_summary,
        )

        reasoning = build_reasoning_payload(
            crop=crop,
            disease=disease,
            confidence=confidence,
            retrieval_context=retrieval_context,
            memory_context=memory_context,
            image_summary=image_summary,
            severity_hint=severity_hint,
        )

        advice = generate_advice(
            crop=crop,
            disease=disease,
            confidence=confidence,
            reasoning=reasoning,
            retrieval_context=retrieval_context,
            memory_context=memory_context,
            image_summary=image_summary,
        )
    except RuntimeError as exc:
        return {
            "session_id": resolved_session_id,
            "memory_key": resolved_memory_key,
            "prediction_mode": mode,
            "crop": crop,
            "disease": disease,
            "confidence": round(confidence, 4),
            "analysis_source": analysis_source,
            "reasoning": ReasoningPayload().model_dump(),
            "retrieval_trace": [],
            "retrieved_context": "",
            "retrieved_documents": [],
            "advice": f"LLM processing failed: {exc}",
            "follow_up_questions": [],
        }

    diagnosis_record = build_diagnosis_record(
        session_id=resolved_memory_key,
        crop=crop,
        disease=disease,
        confidence=confidence,
        reasoning=reasoning,
        analysis_source=analysis_source,
        retrieval_trace=retrieval_trace,
        image_summary=image_summary,
    )
    memory_store.record_diagnosis(resolved_memory_key, diagnosis_record)

    return {
        "session_id": resolved_session_id,
        "memory_key": resolved_memory_key,
        "prediction_mode": mode,
        "crop": crop,
        "disease": disease,
        "confidence": round(confidence, 4),
        "analysis_source": analysis_source,
        "reasoning": reasoning.model_dump(),
        "retrieval_trace": retrieval_trace,
        "retrieved_context": retrieval_context,
        "retrieved_documents": [document.metadata for document in retrieval_documents],
        "advice": clean_json_output(advice),
        "follow_up_questions": reasoning.follow_up_questions,
    }


@app.post("/api/chat")
async def follow_up_chat(payload: ChatRequest = Body(...)):
    session_id = memory_store.ensure_session(payload.session_id)
    session = memory_store.get_session(session_id)
    latest_diagnosis = (session.get("diagnoses") or [])[-1:] or [{}]
    diagnosis_context = latest_diagnosis[0]
    memory_context = memory_store.recent_context(session_id)

    memory_store.record_message(session_id, "user", payload.message, {"type": "follow_up"})

    prompt = f"""
You are continuing a crop diagnosis conversation.

Latest diagnosis:
{json.dumps(diagnosis_context, indent=2, ensure_ascii=False)}

Conversation memory:
{memory_context or 'None'}

User question:
{payload.message}

Answer in markdown. Be specific, practical, and concise.
If the question is about unavailable treatments, offer substitutes and mention confirming local guidance.
If the question suggests severe crop damage, advise urgent local field inspection.
"""

    try:
        answer = invoke_text_llm(prompt, expect_json=False)
    except RuntimeError as exc:
        answer = f"Unable to answer follow-up right now: {exc}"
    memory_store.record_message(session_id, "assistant", answer, {"type": "follow_up_answer"})

    return {
        "session_id": session_id,
        "answer": answer,
    }


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    return memory_store.get_session(session_id)

