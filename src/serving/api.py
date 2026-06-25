"""FastAPI app for routed VLM QA inference."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.routing.task_router import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MULTIMODAL_ROUTER_DIR,
    MultimodalDebertaClipTaskRouter,
    RouterDecision,
    select_task_backend_for_image,
)


app = FastAPI(
    title="Routed VLM QA API",
    version="0.1.0",
    description="Route ChartQA, DocVQA, and TextVQA questions to the best backend.",
)


class PredictResponse(BaseModel):
    """JSON response returned by the /predict endpoint."""

    answer: str
    question: str
    task_type: str
    backend: str
    use_adapter: bool
    adapter: str | None = None
    confidence: float | None = None
    latency_seconds: float


class HealthResponse(BaseModel):
    """JSON response returned by the /health endpoint."""

    status: str
    router_loaded: bool
    model_loaded: bool


def load_router() -> MultimodalDebertaClipTaskRouter | None:
    """Load the multimodal router if its checkpoint exists.

    If the checkpoint is missing, return None. The API then falls back to the
    rule-based router through select_task_backend_for_image(...).
    """
    router_dir = DEFAULT_MULTIMODAL_ROUTER_DIR
    classifier_path = router_dir / "multimodal_logreg.joblib"

    if not classifier_path.exists():
        return None

    return MultimodalDebertaClipTaskRouter.load(router_dir)


@app.on_event("startup")
def startup() -> None:
    """Load long-lived server objects once when the API starts."""
    app.state.router = load_router()

    # Later this becomes:
    # app.state.service = RoutedVLMService(...)
    #
    # For now this API validates routing and request handling without loading
    # the heavy Qwen2.5-VL model.
    app.state.model_loaded = False


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return API readiness information."""
    return HealthResponse(
        status="ok",
        router_loaded=app.state.router is not None,
        model_loaded=bool(app.state.model_loaded),
    )


async def save_upload_to_temp_file(image: UploadFile) -> str:
    """Save an uploaded image to a temporary local path."""
    suffix = Path(image.filename or "image.png").suffix or ".png"

    try:
        content = await image.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read uploaded image.",
        ) from exc

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(content)
        return handle.name


def decision_to_payload(decision: RouterDecision) -> dict[str, Any]:
    """Convert a router decision into JSON-safe metadata."""
    return {
        "task_type": decision.task_type,
        "backend": decision.backend_name,
        "use_adapter": decision.use_adapter,
        "adapter": decision.adapter_name,
        "confidence": decision.confidence,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    question: str = Form(...),
    image: UploadFile = File(...),
    min_confidence: float = Form(DEFAULT_MIN_CONFIDENCE),
) -> PredictResponse:
    """Route one image-question pair and return a prediction payload.

    Current version:
    - saves the uploaded image to a temporary file
    - routes image/question to the selected task backend
    - returns a placeholder answer

    Next version:
    - replace the placeholder with app.state.service.predict(...)
    """
    started_at = time.perf_counter()

    clean_question = question.strip()
    if not clean_question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    image_path = await save_upload_to_temp_file(image)

    decision = select_task_backend_for_image(
        question=clean_question,
        image_path=image_path,
        router=app.state.router,
        min_confidence=min_confidence,
    )

    # Placeholder until Qwen service inference is wired.
    answer = "model not loaded yet"

    latency = time.perf_counter() - started_at
    payload = decision_to_payload(decision)

    return PredictResponse(
        answer=answer,
        question=clean_question,
        task_type=payload["task_type"],
        backend=payload["backend"],
        use_adapter=payload["use_adapter"],
        adapter=payload["adapter"],
        confidence=payload["confidence"],
        latency_seconds=latency,
    )
