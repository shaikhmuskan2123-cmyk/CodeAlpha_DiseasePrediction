from contextlib import asynccontextmanager
from datetime import timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from backend.app.catalog import DISEASES
from backend.app.config import CORS_ORIGINS, HISTORY_ENABLED
from backend.app.database import clear_predictions, initialize_database, list_predictions
from backend.app.schemas import PredictionRequest
from backend.app.services.model_service import (
    dataset_summary,
    get_artifact,
    load_metrics,
    model_path,
    predict,
)
from backend.app.services.report_service import create_prediction_pdf

@asynccontextmanager
async def lifespan(application: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="MediPredict AI API",
    version="0.1.0",
    description="Educational multi-disease machine-learning demonstration; not a diagnostic service.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "MediPredict AI", "api_version": app.version}


@app.get("/api/models")
def models() -> list[dict]:
    output = []
    for disease, details in DISEASES.items():
        ready = model_path(disease).is_file()
        entry = {"id": disease, **details, "available": ready, "features": []}
        if ready:
            try:
                entry["features"] = get_artifact(disease)["metadata"]["features"]
            except (ValueError, OSError, KeyError):
                entry["available"] = False
        output.append(entry)
    return output


def require_disease(disease: str) -> None:
    if disease not in DISEASES:
        raise HTTPException(status_code=404, detail="Unknown disease module.")


@app.get("/api/models/{disease}/metrics")
def metrics(disease: str) -> dict:
    require_disease(disease)
    try:
        return load_metrics(disease)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/datasets/{disease}")
def dataset(disease: str) -> dict:
    require_disease(disease)
    try:
        return dataset_summary(disease)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/predict/{disease}")
def prediction(disease: str, payload: PredictionRequest) -> dict:
    require_disease(disease)
    try:
        persist = HISTORY_ENABLED and (
            HISTORY_ENABLED if payload.persist is None else payload.persist
        )
        return predict(disease, payload.features, persist=persist)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/reports/{disease}")
def report(disease: str, payload: PredictionRequest) -> Response:
    require_disease(disease)
    try:
        pdf, result = create_prediction_pdf(disease, payload.features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    filename = f"medipredict-{disease}-{result['reference'].lower()}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/history")
def history() -> list[dict]:
    return [
        {
            "id": record.id,
            "disease": record.disease,
            "predicted_class": record.predicted_class,
            "positive_score": record.positive_score,
            "created_at": record.created_at.replace(tzinfo=timezone.utc).isoformat(),
        }
        for record in list_predictions()
    ]


@app.delete("/api/history")
def delete_history() -> dict:
    clear_predictions()
    return {"deleted": True}