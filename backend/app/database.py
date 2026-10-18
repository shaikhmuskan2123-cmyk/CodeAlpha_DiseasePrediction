from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from backend.app.config import DATABASE_URL

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    disease: Mapped[str] = mapped_column(String(40), index=True)
    predicted_class: Mapped[str] = mapped_column(String(100))
    positive_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )


def initialize_database() -> None:
    Base.metadata.create_all(engine)


def save_prediction(
    disease: str, predicted_class: str, positive_score: float | None
) -> PredictionRecord:
    with SessionLocal() as session:
        record = PredictionRecord(
            disease=disease,
            predicted_class=predicted_class,
            positive_score=positive_score,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record


def list_predictions() -> list[PredictionRecord]:
    with SessionLocal() as session:
        return list(
            session.scalars(
                select(PredictionRecord).order_by(PredictionRecord.created_at.desc())
            )
        )


def clear_predictions() -> None:
    with SessionLocal() as session:
        session.query(PredictionRecord).delete()
        session.commit()