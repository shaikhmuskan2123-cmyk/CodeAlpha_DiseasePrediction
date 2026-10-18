import pytest
from fastapi.testclient import TestClient
from reportlab.pdfbase.pdfmetrics import stringWidth

from backend.app.main import app
from backend.app.database import list_predictions
from backend.app.services.report_service import DISCLAIMER
from backend.app.services.model_service import get_artifact, predict
from backend.app.services.report_service import create_prediction_pdf


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_unknown_disease_is_rejected():
    with TestClient(app) as client:
        response = client.post("/api/predict/not-a-module", json={"features": {}})
    assert response.status_code == 404


def test_disclaimer_is_present_and_reportlab_can_render_it():
    assert "not a medical diagnosis" in DISCLAIMER
    assert stringWidth(DISCLAIMER, "Helvetica", 9) > 0


def test_models_endpoint_lists_three_independent_modules():
    with TestClient(app) as client:
        response = client.get("/api/models")
    assert response.status_code == 200
    assert {item["id"] for item in response.json()} == {
        "heart",
        "diabetes",
        "breast-cancer",
    }


def _sample_features(disease):
    artifact = get_artifact(disease)
    return {
        feature["name"]: (
            feature["categories"][0]
            if feature["kind"] == "categorical"
            else feature["median"]
        )
        for feature in artifact["metadata"]["features"]
    }


@pytest.mark.parametrize("disease", ["heart", "diabetes", "breast-cancer"])
def test_real_saved_pipeline_predicts_with_validated_features(disease):
    result = predict(disease, _sample_features(disease), persist=False)
    assert result["predicted_class"]
    assert result["model_name"]
    assert result["positive_score"] is None or 0 <= result["positive_score"] <= 1


def test_logistic_explanation_contributions_sum_to_decision_score():
    result = predict("heart", _sample_features("heart"), persist=False)
    explanation = result["explanation"]
    if explanation["scope"] == "this input":
        assert explanation["class_value"] == 1
        assert explanation["intercept"] + explanation["contribution_sum"] == pytest.approx(
            explanation["decision_score"], abs=1e-8
        )
        assert all("contribution" in item for item in explanation["features"])


def test_non_linear_model_explanation_is_explicitly_global():
    result = predict("diabetes", _sample_features("diabetes"), persist=False)
    assert result["explanation"]["scope"] == "global model summary; not this input"


def test_prediction_report_is_generated_in_memory():
    pdf, result = create_prediction_pdf("heart", _sample_features("heart"))
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 1000
    assert result["reference"].startswith("MP-")


@pytest.mark.parametrize(
    ("disease", "expected_rows"),
    [("heart", 303), ("diabetes", 520), ("breast-cancer", 569)],
)
def test_dataset_api_returns_actual_training_records(disease, expected_rows):
    with TestClient(app) as client:
        response = client.get(f"/api/datasets/{disease}")
    assert response.status_code == 200
    assert response.json()["rows"] == expected_rows
    assert len(response.json()["records"]) == expected_rows


def test_prediction_api_rejects_incomplete_features():
    with TestClient(app) as client:
        response = client.post("/api/predict/heart", json={"features": {"age": 50}})
    assert response.status_code == 422


def test_prediction_api_does_not_persist_when_disabled():
    before_ids = {record.id for record in list_predictions()}
    with TestClient(app) as client:
        response = client.post(
            "/api/predict/heart",
            json={"features": _sample_features("heart"), "persist": False},
        )
    assert response.status_code == 200
    assert "history_id" not in response.json()
    assert {record.id for record in list_predictions()} == before_ids


def test_report_endpoint_returns_pdf_download():
    with TestClient(app) as client:
        response = client.post(
            "/api/reports/heart", json={"features": _sample_features("heart")}
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")


def test_history_save_read_and_clear_lifecycle():
    with TestClient(app) as client:
        client.delete("/api/history")
        prediction_response = client.post(
            "/api/predict/heart",
            json={"features": _sample_features("heart"), "persist": True},
        )
        assert prediction_response.status_code == 200
        assert "history_id" in prediction_response.json()
        rows = client.get("/api/history").json()
        assert len(rows) == 1
        assert "features" not in rows[0]
        assert client.delete("/api/history").json() == {"deleted": True}
        assert client.get("/api/history").json() == []