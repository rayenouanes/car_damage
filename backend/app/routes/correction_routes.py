from fastapi import APIRouter, HTTPException, Request

from backend.app.models.schemas import CorrectionRequest


router = APIRouter(prefix="/api", tags=["corrections-error-bank"])


@router.post("/predictions/{prediction_id}/corrections")
def correct_prediction(
    prediction_id: str, correction: CorrectionRequest, request: Request
) -> dict:
    try:
        return request.app.state.correction_service.correct(prediction_id, correction)
    except KeyError:
        raise HTTPException(404, "Prediction introuvable") from None


@router.get("/error-bank")
def list_error_bank(request: Request, limit: int = 500) -> list[dict]:
    return request.app.state.database.list_error_bank(min(max(limit, 1), 2000))

