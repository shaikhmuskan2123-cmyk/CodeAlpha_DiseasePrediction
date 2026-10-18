from datetime import datetime, timezone
from html import escape
from io import BytesIO
from uuid import uuid4

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.app.catalog import DISEASES
from backend.app.services.model_service import load_metrics, predict

DISCLAIMER = (
    "Educational machine learning demonstration only. This output is not a medical diagnosis, "
    "is not clinically validated, and is not a substitute for professional medical advice."
)


def create_prediction_pdf(disease: str, features: dict) -> tuple[bytes, dict]:
    result = predict(disease, features, persist=False)
    metrics = load_metrics(disease)
    reference = f"MP-{uuid4().hex[:10].upper()}"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(buffer, pagesize=letter, title="MediPredict AI report")
    story = [
        Paragraph("MediPredict AI", styles["Title"]),
        Paragraph("Educational model output report", styles["Heading2"]),
        Paragraph(f"Reference: {reference} &nbsp;|&nbsp; Generated: {generated_at}", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(escape(DISEASES[disease]["name"]), styles["Heading2"]),
        Paragraph(f"Model: {escape(result['model_name'])}", styles["BodyText"]),
        Paragraph(f"Dataset: {escape(DISEASES[disease]['dataset'])}", styles["BodyText"]),
        Paragraph(f"Output: {escape(result['predicted_class'])}", styles["BodyText"]),
        Paragraph(
            "Positive-class score: "
            + (f"{result['positive_score']:.4f}" if result["positive_score"] is not None else "Unavailable"),
            styles["BodyText"],
        ),
        Paragraph(escape(result["target_definition"]), styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("Submitted measurements", styles["Heading3"]),
    ]
    rows = [["Feature", "Value"]]
    rows.extend([[escape(str(name)), escape(str(value))] for name, value in features.items()])
    table = Table(rows, repeatRows=1, colWidths=[260, 230])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17343b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d7e2e3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f7f6")]),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([table, Spacer(1, 12), Paragraph("Evaluation context", styles["Heading3"])])
    story.append(Paragraph(f"Evaluation samples: {metrics.get('test_size', 'Unavailable')}", styles["BodyText"]))
    story.append(Paragraph(f"Test ROC-AUC: {metrics.get('roc_auc', 'Unavailable')}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(escape(DISEASES[disease]["population"]), styles["BodyText"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>{escape(DISCLAIMER)}</b>", styles["BodyText"]))
    document.build(story)
    return buffer.getvalue(), {"reference": reference, **result}