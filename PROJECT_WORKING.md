# Project Working Overview

This project can be understood as a generative AI system with retrieval and memory, not just as a crop classifier.

## What The GenAI System Does

The user gives the system a crop image, and the backend turns that image into a grounded natural-language response. The response is not produced by a language model alone. It is assembled from three inputs:

- visual understanding of the image
- retrieved domain knowledge from the project knowledge base
- session memory from earlier interactions

The final result is a diagnosis plus treatment guidance that stays tied to the same conversation.

## Core GenAI Pipeline

1. The user uploads or captures a crop image in the React frontend.
2. The frontend sends the image and session id to the FastAPI backend at `/api/predict`.
3. The backend first creates a prediction signal:
  - in `multimodal_llm` mode, a vision-capable model reads the image directly
  - in `cnn_llm` mode, a crop-specific CNN predicts the disease class
4. The backend then retrieves relevant agronomy passages from FAISS using the crop, disease, confidence, and recent conversation context.
5. The backend converts that retrieved context into a structured reasoning payload.
6. A text LLM turns the reasoning payload and retrieved context into a farmer-facing answer.
7. The frontend displays the result and keeps the session id so the user can ask follow-up questions.
8. When the user asks a follow-up, the frontend sends the same session id to `/api/chat`, and the backend continues the conversation with memory.

## GenAI Building Blocks

### 1. Input Layer

The input is multimodal: an image plus optional crop selection. The session id acts as conversation state.

### 2. Inference Layer

The system supports two inference styles:

- vision-first inference for image understanding
- CNN-based classification for crop-specific prediction

This gives the system a deterministic fallback while still behaving like a GenAI assistant.

### 3. Retrieval Layer

FAISS and `server/kbase.json` provide retrieval-augmented generation. This is the grounding step that keeps the model from answering only from internal parameters.

### 4. Reasoning Layer

The backend builds a structured intermediate object with severity, confidence, intervention priority, evidence, and follow-up questions. This is the system’s internal reasoning layer before the final response is written.

### 5. Generation Layer

The text LLM takes the retrieved context and reasoning object and writes the final answer in natural language. The output is formatted for a user, not just a machine.

### 6. Memory Layer

Session memory stores previous diagnoses and chat turns. That makes the assistant stateful across requests, so the conversation can continue without losing context.

## Agent And Tools

The backend uses a small retrieval agent inside the prediction pipeline. Its job is not to answer the user directly. Instead, it decides how to gather the best supporting context before the final LLM response is written.

The agent works with one main tool:

- `retrieve_crop_knowledge`: searches the FAISS index with a focused query built from the crop, disease, confidence, image summary, and recent session memory.

The agentic flow is simple:

1. Build a retrieval query from the current diagnosis.
2. Call the retrieval tool to fetch relevant documents.
3. Deduplicate and format the retrieved passages.
4. Pass that context to the reasoning and generation steps.

If the optional agent layer cannot load, the backend still works because it falls back to direct FAISS search. That means the system keeps producing grounded answers even when the agent framework is unavailable.

From a GenAI perspective, the agent is the orchestration layer. It helps the model decide what context to retrieve, while the LLM itself handles reasoning and response writing.

## How This Looks As A GenAI Product

From a GenAI point of view, the system is doing four things:

- understanding the image
- grounding the answer in a knowledge base
- generating a useful explanation
- remembering the prior conversation

That is why the system behaves like an AI assistant rather than a single-shot classifier.

## Short Summary

The project is a grounded multimodal GenAI pipeline: image in, retrieval and memory in the middle, natural-language advice out.
